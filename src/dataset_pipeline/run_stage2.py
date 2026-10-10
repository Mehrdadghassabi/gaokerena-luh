"""
Stage 2 of the LUH dataset pipeline -- resumable, crash-safe runner.

Adapted from luh/cli/generate_dataset/run_extract_verify_claims.py, applied to
our Question/Response xlsx files. Produces, per row:
    claims, verified, uncertainty_labels
plus the input_ids the training path needs.

DURABILITY -- nothing is lost on interruption:
  * Rows are processed in small chunks; each finished row is appended to
    rows.jsonl and fsync'd before the next chunk starts.
  * Every OpenRouter response is cached on disk by (model, prompt). So even an
    interruption *mid-row* costs no money: the resumed run replays the cached
    answers instantly instead of re-billing them.
  * Cumulative token usage is appended to usage.jsonl, and a human-readable
    snapshot is rewritten to progress.json after every chunk.
  * Failed rows go to errors.jsonl and the run continues, so one bad row cannot
    kill a long job.

RESUME: just run the same command again. Completed row indices are read back
from rows.jsonl and skipped. Use --restart to deliberately start over.

Pipeline settings baked in (established earlier in this work):
  * Persian prompts (prompts_fa.py) so claims stay in Persian.
  * Heading/fragment sentences skipped before extraction (no API call, no junk).
  * Reasoning disabled -- deepseek-v4-flash-0731 otherwise spends its whole
    completion budget on hidden chain-of-thought, which is ~10x slower and can
    return empty content.

Usage:
    # data-1 was built on OpenRouter (the default provider):
    HF_TOKEN=... python run_stage2.py --dataset LUH-data-1-selected --rows 20

    # official DeepSeek API instead (needs DEEPSEEK_API_KEY):
    HF_TOKEN=... python run_stage2.py \
        --provider deepseek --dataset LUH-data-2-selected --rows 20

    # then, once all rows are done, materialise an HF dataset for training:
    python run_stage2.py --dataset LUH-data-1-selected --finalize
"""
import argparse
import importlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import DATA_DIR, WORK_DIR, read_api_key

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
TOKENIZER_NAME = "CohereLabs/aya-expanse-8b"
LANGUAGE = "fa"

# --------------------------------------------------------------------------
# providers
#
# Both talk the OpenAI wire protocol, but they differ in two ways that matter
# and are NOT interchangeable:
#
#  1. Turning chain-of-thought off. OpenRouter takes `reasoning={"enabled":
#     False}`; the official DeepSeek API SILENTLY IGNORES that key (verified --
#     reasoning_content still comes back) and wants
#     `thinking={"type": "disabled"}` instead. Sending the wrong one costs no
#     error, just a run with reasoning left on: ~10x slower, and it can exhaust
#     the completion budget and return content=None.
#  2. Price. Official DeepSeek is far more expensive than OpenRouter was for
#     the same model, and its rate halves outside peak hours.
# --------------------------------------------------------------------------
PROVIDERS = {
    "openrouter": {
        "base_url": OPENROUTER_BASE_URL,
        "key_env": "OPENROUTER_API_KEY",
        "key_file": "OPENROUTER_KEY.txt",
        "default_model": DEFAULT_MODEL,
        "no_think": {"reasoning": {"enabled": False}},
        # Verified from OpenRouter's /models at the time of the data-1 run.
        "price_in": 0.09 / 1_000_000,
        "price_in_cached": 0.09 / 1_000_000,   # no separate cache tier used
        "price_out": 0.18 / 1_000_000,
        "offpeak_discount": 1.0,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "key_file": "DEEPSEEK_KEY.txt",
        # The official API offers only the rolling `deepseek-v4-flash`; the
        # `-0731` snapshot used for data-1 is not selectable here.
        "default_model": "deepseek-v4-flash",
        "no_think": {"thinking": {"type": "disabled"}},
        # api-docs.deepseek.com/quick_start/pricing -- PEAK rates per 1M.
        "price_in": 0.44 / 1_000_000,
        "price_in_cached": 0.014 / 1_000_000,
        "price_out": 1.32 / 1_000_000,
        # Off-peak is 50% of peak. Peak = 01:00-04:00 and 06:00-10:00 UTC,
        # Mon-Fri; everything else is off-peak.
        "offpeak_discount": 0.5,
    },
}

# Populated from the chosen provider in main(); module-level so the usage
# tracker and cost_of() can see them.
PRICE_IN = PROVIDERS["openrouter"]["price_in"]
PRICE_IN_CACHED = PROVIDERS["openrouter"]["price_in_cached"]
PRICE_OUT = PROVIDERS["openrouter"]["price_out"]
NO_THINK_BODY = PROVIDERS["openrouter"]["no_think"]
OFFPEAK_DISCOUNT = 1.0


def _is_peak_now():
    """Peak = 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri (DeepSeek pricing)."""
    now = time.gmtime()
    if now.tm_wday >= 5:  # Sat/Sun
        return False
    h = now.tm_hour
    return (1 <= h < 4) or (6 <= h < 10)

# A "sentence" shorter than this (after stripping markdown) is a heading/label
# with no verifiable fact. See run_stage2_sample.py for the calibration.
MIN_SENTENCE_CHARS = 40

import openai  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from datasets import Dataset  # noqa: E402

from lm_polygraph.stat_calculators import ClaimsExtractor  # noqa: E402
from lm_polygraph.utils.openai_chat import OpenAIChat  # noqa: E402
from lm_polygraph.generation_metrics.openai_fact_check import OpenAIFactCheck  # noqa: E402
from lm_polygraph.utils.model import WhiteboxModel  # noqa: E402
from lm_polygraph.stat_calculators.claim_level_prompts import (  # noqa: E402
    CLAIM_EXTRACTION_PROMPTS,
    MATCHING_PROMPTS,
    OPENAI_FACT_CHECK_PROMPTS,
    OPENAI_FACT_CHECK_SUMMARIZE_PROMPT,
)


# --------------------------------------------------------------------------
# Verbatim from luh/cli/generate_dataset/run_extract_verify_claims.py in the
# llm-uncertainty-head repo, copied so this folder does not depend on `luh`.
# --------------------------------------------------------------------------
def generate_targets(dataset, reply_tokens_all, tokenizer):
    targets = []

    all_claims = dataset["claims"]
    all_verified = dataset["verified"]
    all_questions = dataset["question"]
    all_input_ids = dataset["input_ids"]

    all_tokenized_questions = tokenizer.apply_chat_template(
        [[{"role": "user", "content": all_questions[idx]}] for idx in range(len(dataset))],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
    )["input_ids"]

    for idx in range(len(dataset)):
        reply_tokens = reply_tokens_all[idx]
        claims = all_claims[idx]
        verified = all_verified[idx]
        input_ids = all_input_ids[idx]

        target = [-100.0] * len(reply_tokens)
        for claim, label in zip(claims, verified):
            for t in claim["aligned_token_ids"]:
                if not np.isnan(label):
                    target[t] = float(label == 1.0)

        encoded_question = all_tokenized_questions[idx]
        final_target = [-100] * len(encoded_question)
        curr_target = 0
        for i in range(len(encoded_question), len(input_ids)):
            if input_ids[i] in tokenizer.all_special_ids:
                final_target.append(-100)
            else:
                final_target.append(target[curr_target])
                curr_target += 1

        assert len(final_target) == len(input_ids)
        targets.append(final_target)

    return targets


def extract_tokens_of_reply(dataset, tokenizer):
    inpt_texts = dataset["question"]
    question_tokens = tokenizer.apply_chat_template(
        [[{"role": "user", "content": e}] for e in inpt_texts],
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"]
    greedy_tokens = []
    inpt_ids = dataset["input_ids"]
    for i in range(len(dataset)):
        tokens_with_special = inpt_ids[i][len(question_tokens[i]):]
        greedy_tokens.append(
            [t for t in tokens_with_special if t not in tokenizer.all_special_ids]
        )

    return greedy_tokens


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
def get_prompts(language):
    """The upstream repo's get_prompts() indexes the prompt dicts by *module*
    name ("benchmark_multilingual.fa.prompts"), raising a KeyError that a bare
    `except: continue` swallows -- so per-language prompts never load there.
    Merge them properly instead. prompts_fa.py is a copy of that repo's
    benchmark_multilingual/fa/prompts.py."""
    prompts = {
        "extraction": dict(CLAIM_EXTRACTION_PROMPTS),
        "matching": dict(MATCHING_PROMPTS),
        "fact_check": dict(OPENAI_FACT_CHECK_PROMPTS),
        "summarize": dict(OPENAI_FACT_CHECK_SUMMARIZE_PROMPT),
    }

    mod = importlib.import_module(f"prompts_{language}")
    prompts["extraction"].update(mod.CLAIM_EXTRACTION_PROMPTS)
    prompts["matching"].update(mod.MATCHING_PROMPTS)
    prompts["fact_check"].update(mod.OPENAI_FACT_CHECK_PROMPTS)
    prompts["summarize"].update(mod.OPENAI_FACT_CHECK_SUMMARIZE_PROMPT)

    for key in prompts:
        if language not in prompts[key]:
            raise KeyError(f"no '{language}' prompt for stage '{key}'")
    return prompts


# --------------------------------------------------------------------------
# claim extraction with heading/fragment filtering
# --------------------------------------------------------------------------
_MD_STRIP = re.compile(r"[*_#`>‌]+")


def is_extractable_sentence(sent):
    return len(_MD_STRIP.sub("", sent).strip()) >= MIN_SENTENCE_CHARS


class FilteredClaimsExtractor(ClaimsExtractor):
    """Skips heading/fragment sentences.

    Filters the (text, tokens, position) triples from the parent's
    split_to_sentences. Positions are absolute token offsets into the reply and
    the parent derives n_sents from len(sent_texts), so dropping whole triples
    keeps claim->token alignment correct.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_skipped = 0
        self.n_kept = 0

    def split_to_sentences(self, text, tokens, tokenizer):
        texts, toks, positions = super().split_to_sentences(text, tokens, tokenizer)
        keep = [
            (s, tk, p) for s, tk, p in zip(texts, toks, positions)
            if is_extractable_sentence(s)
        ]
        self.n_skipped += len(texts) - len(keep)
        self.n_kept += len(keep)
        if not keep:
            return [], [], []
        a, b, c = zip(*keep)
        return list(a), list(b), list(c)


# --------------------------------------------------------------------------
# usage tracking + reasoning-off patch
# --------------------------------------------------------------------------
usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "cache_hit_tokens": 0, "cache_miss_tokens": 0,
         "reasoning_tokens": 0, "usd": 0.0}
_orig_create = openai.resources.chat.completions.Completions.create
_usage_bar = None


def _patched_create(self, *args, **kwargs):
    # Force reasoning off. deepseek-v4-flash is a reasoning model: left on, it
    # spends most of its completion budget on hidden chain-of-thought before the
    # answer -- ~10x slower here, and it can exhaust the budget entirely,
    # returning content=None and crashing lm_polygraph's OpenAIChat.ask.
    # NO_THINK_BODY is provider-specific -- see PROVIDERS above.
    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.update(NO_THINK_BODY)
    kwargs["extra_body"] = extra_body

    resp = _orig_create(self, *args, **kwargs)

    if resp.choices and resp.choices[0].message.content is None:
        resp.choices[0].message.content = ""

    usage["calls"] += 1
    u = getattr(resp, "usage", None)
    if u is not None:
        p_tok = u.prompt_tokens or 0
        c_tok = u.completion_tokens or 0
        usage["prompt_tokens"] += p_tok
        usage["completion_tokens"] += c_tok

        # DeepSeek splits prompt tokens into cache hit/miss and prices them very
        # differently ($0.014 vs $0.44 per 1M at peak). OpenRouter omits these.
        hit = getattr(u, "prompt_cache_hit_tokens", None)
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        if hit is None and miss is None:
            hit, miss = 0, p_tok
        else:
            hit = hit or 0
            miss = miss if miss is not None else max(0, p_tok - hit)
        usage["cache_hit_tokens"] += hit
        usage["cache_miss_tokens"] += miss

        # Guard rail: if this ever climbs, reasoning is NOT actually off and the
        # provider is ignoring NO_THINK_BODY.
        det = getattr(u, "completion_tokens_details", None)
        if det is not None:
            usage["reasoning_tokens"] += getattr(det, "reasoning_tokens", None) or 0

        rate = OFFPEAK_DISCOUNT if not _is_peak_now() else 1.0
        usage["usd"] += rate * (miss * PRICE_IN
                                + hit * PRICE_IN_CACHED
                                + c_tok * PRICE_OUT)

    if _usage_bar is not None:
        _usage_bar.update(1)
        _usage_bar.set_postfix(
            in_tok=usage["prompt_tokens"],
            out_tok=usage["completion_tokens"],
            usd=f"{cost_of(usage):.4f}",
        )
    return resp


openai.resources.chat.completions.Completions.create = _patched_create


def _is_out_of_credit(exc):
    """True if this failure is 'the account ran out of money', which is not a
    per-row problem: every remaining row would fail the same way. Better to stop
    cleanly than to churn the whole todo list into errors.jsonl."""
    s = f"{type(exc).__name__}: {exc}".lower()
    if getattr(exc, "status_code", None) in (402,):
        return True
    return any(m in s for m in (
        "insufficient balance", "insufficient_quota", "insufficient credits",
        "exceeded your current quota", "402", "payment required",
        "requires more credits",
    ))


def merge_usage(prior, cur):
    """Sum every counter, so cache-tier splits, reasoning tokens and the actual
    accrued USD all survive a resume instead of being silently recomputed."""
    return {k: (prior.get(k, 0) or 0) + (cur.get(k, 0) or 0)
            for k in set(prior) | set(cur)}


def cost_of(u):
    """Actual accrued spend when available (it accounts for cache tiers and the
    peak/off-peak window at the moment of each call); otherwise a flat estimate,
    which is what old usage.jsonl records from the OpenRouter run fall back to."""
    if u.get("usd"):
        return u["usd"]
    return (u.get("cache_miss_tokens", u.get("prompt_tokens", 0)) * PRICE_IN
            + u.get("cache_hit_tokens", 0) * PRICE_IN_CACHED
            + u.get("completion_tokens", 0) * PRICE_OUT)


# --------------------------------------------------------------------------
# durable IO
# --------------------------------------------------------------------------
def append_jsonl(path, obj):
    """Append one record and force it to disk, so an interruption immediately
    after cannot lose it."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def repair_jsonl(path):
    """Drop an incomplete trailing record left by a hard kill mid-write.

    Necessary because a partial line has no terminating newline, so the next
    append would concatenate onto it and corrupt an otherwise-good record too
    (i.e. the interruption would cost us a second row). Truncating back to the
    last complete newline makes appends safe again.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0
        # find the last newline and cut everything after it
        f.seek(0)
        data = f.read()
        cut = data.rfind(b"\n") + 1  # 0 if there is no newline at all
        dropped = len(data) - cut
        f.seek(cut)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())
    print(f"  repaired {os.path.basename(path)}: dropped {dropped} bytes of an "
          f"incomplete trailing record (interrupted mid-write)")
    return dropped


def read_jsonl(path):
    """Tolerates a truncated final line, which is what a hard kill mid-write
    leaves behind."""
    if not os.path.exists(path):
        return []
    out, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  note: skipped {bad} unparseable line(s) in {os.path.basename(path)} "
              f"(expected after a hard interruption mid-write)")
    return out


def nan_to_none(xs):
    return [None if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in xs]


def none_to_nan(xs):
    return [float("nan") if x is None else x for x in xs]


def build_input_ids(tokenizer, question, reply):
    """Approximate the generation-time input_ids: chat-templated question +
    reply tokens + eos. We don't have the original generation token ids."""
    question_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
        )["input_ids"]
    )
    reply_ids = list(tokenizer(reply, add_special_tokens=False)["input_ids"])
    eos = tokenizer.eos_token_id
    return question_ids + reply_ids + ([eos] if eos is not None else [])


# --------------------------------------------------------------------------
# per-chunk processing
# --------------------------------------------------------------------------
def process_chunk(indices, df, tokenizer, extractor, fact_checker, model):
    questions = [str(df["Question"].iloc[i]) for i in indices]
    replies = [str(df["Response"].iloc[i]) for i in indices]
    input_ids_list = [build_input_ids(tokenizer, q, r) for q, r in zip(questions, replies)]

    base = Dataset.from_dict(
        {"question": questions, "reply": replies, "input_ids": input_ids_list}
    )

    greedy_tokens = extract_tokens_of_reply(base, tokenizer)
    deps = {"greedy_texts": base["reply"], "greedy_tokens": greedy_tokens}

    claims_raw = extractor(deps, base["question"], model=model)
    claims = [list(c) for c in claims_raw["claims"]]

    verified = fact_checker({"input_texts": base["question"], "claims": claims}, None)

    result = base.to_dict()
    result["claims"] = [[c.__dict__ for c in row] for row in claims]
    result["verified"] = verified
    labels = generate_targets(Dataset.from_dict(result), greedy_tokens, tokenizer)

    records = []
    for k, row_idx in enumerate(indices):
        records.append(
            {
                "row_index": int(row_idx),
                "question": questions[k],
                "reply": replies[k],
                "input_ids": [int(t) for t in input_ids_list[k]],
                "claims": result["claims"][k],
                "verified": nan_to_none([float(v) for v in verified[k]]),
                "uncertainty_labels": [float(v) for v in labels[k]],
            }
        )
    return records


# --------------------------------------------------------------------------
# finalize: rows.jsonl -> HF dataset
# --------------------------------------------------------------------------
def finalize(out_dir, dataset_name):
    rows_path = os.path.join(out_dir, "rows.jsonl")
    records = read_jsonl(rows_path)
    if not records:
        print(f"Nothing to finalize: no rows have been generated yet.")
        print(f"  looked in: {rows_path}")
        print(f"\n--finalize only packages rows that a previous run already produced;")
        print(f"it does not generate anything itself. Run the generation step first")
        print(f"(same command WITHOUT --finalize), e.g.:")
        print(f"\n  python run_stage2.py --dataset {dataset_name} --rows 20\n")
        print(f"then re-run with --finalize.")
        return 1

    records.sort(key=lambda r: r["row_index"])
    seen, dedup = set(), []
    for r in records:
        if r["row_index"] in seen:
            continue
        seen.add(r["row_index"])
        dedup.append(r)

    ds = Dataset.from_dict(
        {
            "question": [r["question"] for r in dedup],
            "reply": [r["reply"] for r in dedup],
            "input_ids": [r["input_ids"] for r in dedup],
            "claims": [r["claims"] for r in dedup],
            "verified": [none_to_nan(r["verified"]) for r in dedup],
            "uncertainty_labels": [r["uncertainty_labels"] for r in dedup],
        }
    )
    save_path = os.path.join(out_dir, "hf_dataset")
    ds.save_to_disk(save_path)

    # Also emit the flat JSON shape that validate_dataset.py /
    # compare_with_reference.py consume, so the output can be checked directly.
    # NOTE: written with real NaN (json's non-standard but Python-round-trippable
    # allow_nan), NOT null. The repo's claim collator does np.isnan(e) on every
    # `verified` value, which raises TypeError on None -- so nulls here would
    # crash training. rows.jsonl deliberately uses strict JSON with null for
    # portability; this file converts back.
    json_path = os.path.join(out_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "question": [r["question"] for r in dedup],
                "reply": [r["reply"] for r in dedup],
                "input_ids": [r["input_ids"] for r in dedup],
                "claims": [r["claims"] for r in dedup],
                "verified": [none_to_nan(r["verified"]) for r in dedup],
                "uncertainty_labels": [r["uncertainty_labels"] for r in dedup],
            },
            f, ensure_ascii=False, allow_nan=True,
        )

    n_claims = sum(len(r["claims"]) for r in dedup)
    print(f"Finalized {len(dedup)} rows ({n_claims} claims)")
    print(f"  HF dataset : {save_path}")
    print(f"  flat JSON  : {json_path}")
    print(f"\nnext: python build_hf_splits.py --dataset {dataset_name}")
    return 0


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="LUH-data-1-selected",
                   help="basename of the xlsx in LUH_DATA_DIR (default: %(default)s)")
    p.add_argument("--rows", type=int, default=20,
                   help="how many rows to process, from --start (default: %(default)s)")
    p.add_argument("--start", type=int, default=0, help="first row index (default: 0)")
    p.add_argument("--chunk-size", type=int, default=4,
                   help="rows per checkpoint; smaller = more frequent saves, "
                        "larger = better thread saturation (default: %(default)s)")
    p.add_argument("--threads", type=int, default=16,
                   help="concurrent API calls (default: %(default)s)")
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="openrouter",
                   help="API provider; sets base_url, key file, default model, "
                        "prices and the reasoning-off flag (default: %(default)s)")
    p.add_argument("--model", default=None,
                   help="override the provider's default model")
    p.add_argument("--out-dir", default=None,
                   help="default: <LUH_WORK_DIR>/stage2/<dataset>")
    p.add_argument("--cache-dir", default=None,
                   help="API response cache; point at an earlier run's api_cache "
                        "to replay it for free (default: <out-dir>/api_cache)")
    p.add_argument("--restart", action="store_true",
                   help="ignore existing progress and start over (does NOT clear "
                        "the API cache, so replays are still free)")
    p.add_argument("--finalize", action="store_true",
                   help="build the HF dataset from rows.jsonl and exit")
    return p.parse_args()


def main():
    global _usage_bar
    global PRICE_IN, PRICE_IN_CACHED, PRICE_OUT, NO_THINK_BODY, OFFPEAK_DISCOUNT
    args = parse_args()

    prov = PROVIDERS[args.provider]
    PRICE_IN = prov["price_in"]
    PRICE_IN_CACHED = prov["price_in_cached"]
    PRICE_OUT = prov["price_out"]
    NO_THINK_BODY = prov["no_think"]
    OFFPEAK_DISCOUNT = prov["offpeak_discount"]
    # NB: not `model` -- that name is rebound to the WhiteboxModel further down.
    api_model = args.model or prov["default_model"]

    out_dir = args.out_dir or os.path.join(WORK_DIR, "stage2", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.finalize:
        return finalize(out_dir, args.dataset)

    hf_token = os.environ.get("HF_TOKEN")
    os.environ["OPENAI_API_KEY"] = read_api_key(prov["key_env"], prov["key_file"])

    rows_path = os.path.join(out_dir, "rows.jsonl")
    usage_path = os.path.join(out_dir, "usage.jsonl")
    errors_path = os.path.join(out_dir, "errors.jsonl")
    # Which judge labelled which rows. Kept OUT of rows.jsonl so the finalized HF
    # dataset keeps exactly data-1's column set; this run is expected to span two
    # providers, and that provenance must not be lost.
    judge_path = os.path.join(out_dir, "judge.jsonl")
    progress_path = os.path.join(out_dir, "progress.json")

    if args.restart:
        for p in (rows_path, usage_path, errors_path, judge_path, progress_path):
            if os.path.exists(p):
                os.rename(p, p + ".bak")
        print("--restart: moved previous progress files to *.bak")

    xlsx = os.path.join(DATA_DIR, f"{args.dataset}.xlsx")
    df = pd.read_excel(xlsx)

    # Heal any partial trailing record before we append to these files.
    for p in (rows_path, usage_path, errors_path, judge_path):
        repair_jsonl(p)

    wanted = list(range(args.start, min(args.start + args.rows, len(df))))
    done_records = read_jsonl(rows_path)
    done = {r["row_index"] for r in done_records}
    todo = [i for i in wanted if i not in done]

    # restore cumulative usage so reported totals span resumes
    prior = read_jsonl(usage_path)
    prior_usage = prior[-1]["cumulative"] if prior else {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "cache_hit_tokens": 0, "cache_miss_tokens": 0,
        "reasoning_tokens": 0, "usd": 0.0
    }

    print("=" * 78)
    print(f"Stage 2  |  dataset={args.dataset}  model={api_model}")
    print(f"provider: {args.provider}  base_url={prov['base_url']}  "
          f"window={'PEAK' if _is_peak_now() else 'off-peak'}")
    print(f"out_dir : {out_dir}")
    print(f"rows    : requested {len(wanted)} (idx {args.start}..{args.start + len(wanted) - 1})"
          f"  already done {len(done & set(wanted))}  to do {len(todo)}")
    print(f"settings: lang={LANGUAGE} threads={args.threads} chunk={args.chunk_size} "
          f"min_sent_chars={MIN_SENTENCE_CHARS} reasoning=off")
    if prior_usage["calls"]:
        print(f"resuming: prior usage {prior_usage['calls']} calls, "
              f"${cost_of(prior_usage):.4f}")
    print("=" * 78)

    if not todo:
        print("\nAll requested rows already complete. "
              "Run with --finalize to build the HF dataset.")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=hf_token)
    prompts = get_prompts(LANGUAGE)

    cache_path = args.cache_dir or os.path.join(out_dir, "api_cache")
    os.makedirs(cache_path, exist_ok=True)

    chat = OpenAIChat(openai_model=api_model, base_url=prov["base_url"],
                      cache_path=cache_path)
    extractor = FilteredClaimsExtractor(
        chat, language=LANGUAGE, progress_bar=False,
        extraction_prompts=prompts["extraction"],
        matching_prompts=prompts["matching"], n_threads=args.threads,
    )
    fact_checker = OpenAIFactCheck(
        llm_url=prov["base_url"], openai_model=api_model, language=LANGUAGE,
        cache_path=cache_path, progress_bar=False,
        fact_check_prompts=prompts["fact_check"],
        fact_check_summarize_prompt=prompts["summarize"], n_threads=args.threads,
    )
    model = WhiteboxModel(model=None, tokenizer=tokenizer)

    chunks = [todo[i:i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]

    _usage_bar = tqdm(desc="API calls", unit="call", position=1,
                      leave=True, dynamic_ncols=True)
    row_bar = tqdm(total=len(todo), desc="rows", unit="row", position=0,
                   leave=True, dynamic_ncols=True)

    t_start = time.time()
    n_ok = n_fail = 0
    out_of_credit = False
    try:
        for chunk in chunks:
            t0 = time.time()
            try:
                records = process_chunk(chunk, df, tokenizer, extractor,
                                        fact_checker, model)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 - one bad chunk must not kill the run
                append_jsonl(errors_path, {
                    "row_indices": [int(i) for i in chunk],
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_s": round(time.time() - t0, 1),
                })
                n_fail += len(chunk)
                row_bar.update(len(chunk))
                if _is_out_of_credit(e):
                    out_of_credit = True
                    print(f"\n\n!! {args.provider} is out of credit -- stopping cleanly "
                          f"after {n_ok} rows.\n"
                          f"   Nothing is lost. Top up and rerun the same command, or "
                          f"switch provider to finish the remaining rows:\n"
                          f"   --provider openrouter --model deepseek/deepseek-v4-flash-0731\n")
                    break
                tqdm.write(f"  !! chunk {chunk[0]}..{chunk[-1]} failed: "
                           f"{type(e).__name__}: {e}  (logged, continuing)")
                continue

            for rec in records:
                append_jsonl(rows_path, rec)
            n_ok += len(records)
            append_jsonl(judge_path, {
                "row_indices": [int(r["row_index"]) for r in records],
                "provider": args.provider,
                "model": api_model,
            })

            cum = merge_usage(prior_usage, usage)
            append_jsonl(usage_path, {
                "row_indices": [int(i) for i in chunk],
                "elapsed_s": round(time.time() - t0, 1),
                "this_session": dict(usage),
                "cumulative": cum,
                "cumulative_usd": round(cost_of(cum), 6),
            })

            n_done_total = len(done) + n_ok
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({
                    "dataset": args.dataset,
                    "provider": args.provider,
                    "model": api_model,
                    "rows_completed_total": n_done_total,
                    "rows_completed_this_session": n_ok,
                    "rows_failed_this_session": n_fail,
                    "requested_range": [args.start, args.start + len(wanted) - 1],
                    "cumulative_usage": cum,
                    "cumulative_usd": round(cost_of(cum), 6),
                    "sentences_kept": extractor.n_kept,
                    "sentences_skipped_as_heading": extractor.n_skipped,
                    "session_elapsed_s": round(time.time() - t_start, 1),
                }, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            row_bar.update(len(records))
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted. Progress is saved -- re-run the same command "
                   "to resume; cached API responses make the replay free.")
    finally:
        row_bar.close()
        _usage_bar.close()

    elapsed = time.time() - t_start
    cum = merge_usage(prior_usage, usage)

    print("\n" + "=" * 78)
    print(f"rows ok this session : {n_ok}")
    if n_fail:
        print(f"rows FAILED          : {n_fail}   (see {errors_path})")
    print(f"total rows on disk   : {len(done) + n_ok}")
    print(f"sentences kept/skipped: {extractor.n_kept} / {extractor.n_skipped}")
    print(f"session elapsed      : {elapsed / 60:.1f} min"
          + (f"  ({elapsed / n_ok:.1f}s/row)" if n_ok else ""))
    print(f"session tokens       : {usage['prompt_tokens']:,} in / "
          f"{usage['completion_tokens']:,} out over {usage['calls']:,} calls")
    print(f"  prompt cache       : {usage['cache_hit_tokens']:,} hit / "
          f"{usage['cache_miss_tokens']:,} miss")
    if usage["reasoning_tokens"]:
        print(f"  !! REASONING ON    : {usage['reasoning_tokens']:,} reasoning tokens "
              f"billed -- provider ignored {NO_THINK_BODY}. Stop and fix; this is "
              f"~10x slower and can return empty content.")
    judged = read_jsonl(judge_path)
    judges = sorted({r["model"] for r in judged if "model" in r})
    print(f"cumulative cost      : ${cost_of(cum):.4f}"
          + ("   (spans multiple providers)" if len(judges) > 1 else ""))
    if len(judges) > 1:
        counts = {m: sum(len(r["row_indices"]) for r in judged if r.get("model") == m)
                  for m in judges}
        print("judge models         : "
              + ", ".join(f"{m} ({n} rows)" for m, n in counts.items())
              + "   <- mixed labelling; see judge.jsonl")
    if out_of_credit:
        print(f"STOPPED: {args.provider} out of credit at {len(done) + n_ok}/{len(wanted)} rows.")
    if n_ok:
        per_row = cost_of(usage) / n_ok
        print(f"\nper-row: ${per_row:.5f}, {elapsed / n_ok:.1f}s")
        for n in (1600, 3200):
            print(f"  -> {n} rows: ~${per_row * n:.2f}, "
                  f"~{elapsed / n_ok * n / 3600:.0f} h at threads={args.threads}")
    print(f"\nfiles: rows.jsonl  usage.jsonl  progress.json"
          + ("  errors.jsonl" if n_fail else ""))
    print(f"next : same command to resume, or --finalize to build the HF dataset")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
