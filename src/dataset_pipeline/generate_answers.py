"""
Stage 1 of the LUH dataset pipeline: answer every question with one model.

RECONSTRUCTED. The script that actually produced data/LUH-data-{1,2} was run
outside this repo (likely Colab) and was never saved. This rebuilds it from the
spec in temp/generate_answers.md -- the plain-HF path of upstream
luh/cli/generate_dataset/run_generate_texts.py, greedy decoding -- and from the
shape of the files it produced:
  * <name>/processed_chunk_NNN.xlsx, 100 rows each, columns Question/Response
  * <name>.xlsx, all chunks concatenated
  * max_new_tokens=1024, NOT the 256 in the spec: the existing responses run to
    a median of ~700 aya tokens and cap at exactly 1024.
Outputs from this script will therefore be close to, but not guaranteed
byte-identical with, the existing files.

Resumable: finished chunk files are skipped, so re-run the same command after
an interruption.

    # the two existing datasets (same questions, two models; local GPU):
    python generate_answers.py --name LUH-data-1 --model CohereLabs/aya-expanse-8b
    python generate_answers.py --name LUH-data-2 --model gaokerena/gaokerena-r1.0

    # any model OpenRouter hosts (needs OPENROUTER_API_KEY, e.g. in .env):
    python generate_answers.py --backend openrouter --name LUH-data-3 \
        --model cohere/command-r7b-12-2024

OpenRouter does NOT host aya-expanse-8b or gaokerena-r1.0, so the openrouter
backend cannot regenerate data-1/-2; it is for answering with other models.
Its replies carry no token ids, which is fine: stage 2 re-tokenises them.

Questions default to the Question column of data/LUH-data-1.xlsx -- the only
copy of the 3200-question list in this folder.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from paths import DATA_DIR, read_api_key

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GENERATION = {
    "do_sample": False,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.0,
}


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_reply(model, tokenizer, generation_config, question, max_new_tokens):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        num_return_sequences=1,
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        **{**GENERATION, "max_new_tokens": max_new_tokens},
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def make_hf_answerer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    hf_token = os.environ.get("HF_TOKEN")
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=hf_token, device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    generation_config = GenerationConfig.from_pretrained(args.model, token=hf_token)
    print(f"model {args.model} on {device}")

    def answer(questions):
        return [generate_reply(model, tokenizer, generation_config, q,
                               args.max_new_tokens)
                for q in tqdm(questions, leave=False)]
    return answer


def make_openrouter_answerer(args):
    import openai

    client = openai.OpenAI(base_url=OPENROUTER_BASE_URL,
                           api_key=read_api_key("OPENROUTER_API_KEY", "OPENROUTER_KEY.txt"),
                           max_retries=5)
    usage = {"in": 0, "out": 0}
    print(f"model {args.model} via OpenRouter, {args.threads} threads")

    def ask(question):
        # temperature 0 is OpenRouter's closest equivalent to greedy decoding
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": question}],
            temperature=0,
            max_tokens=args.max_new_tokens,
        )
        if resp.usage:
            usage["in"] += resp.usage.prompt_tokens or 0
            usage["out"] += resp.usage.completion_tokens or 0
        return resp.choices[0].message.content or ""

    def answer(questions):
        with ThreadPoolExecutor(args.threads) as pool:
            replies = list(tqdm(pool.map(ask, questions), total=len(questions), leave=False))
        tqdm.write(f"  tokens so far: {usage['in']:,} in / {usage['out']:,} out")
        return replies
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="output name, e.g. LUH-data-1 -> data/LUH-data-1/ + data/LUH-data-1.xlsx")
    ap.add_argument("--model", required=True,
                    help="HF model id / local path, or an OpenRouter model id")
    ap.add_argument("--backend", choices=["hf", "openrouter"], default="hf",
                    help="hf = local transformers (default); openrouter = API")
    ap.add_argument("--threads", type=int, default=8,
                    help="concurrent requests for --backend openrouter")
    ap.add_argument("--questions", default=os.path.join(DATA_DIR, "LUH-data-1.xlsx"),
                    help="xlsx with a Question column (default: %(default)s)")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=GENERATION["max_new_tokens"])
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N questions (for a smoke test)")
    args = ap.parse_args()

    questions = pd.read_excel(args.questions)["Question"].astype(str).tolist()
    if args.limit:
        questions = questions[:args.limit]

    chunk_dir = os.path.join(DATA_DIR, args.name)
    os.makedirs(chunk_dir, exist_ok=True)
    chunks = [questions[i:i + args.chunk_size]
              for i in range(0, len(questions), args.chunk_size)]
    todo = [k for k in range(len(chunks))
            if not os.path.exists(os.path.join(chunk_dir, f"processed_chunk_{k + 1:03d}.xlsx"))]
    print(f"{len(questions)} questions, {len(chunks)} chunks, {len(todo)} to do "
          f"-> {chunk_dir}")

    if todo:
        answer = (make_openrouter_answerer if args.backend == "openrouter"
                  else make_hf_answerer)(args)
        for k in tqdm(todo, desc="chunks"):
            replies = answer(chunks[k])
            path = os.path.join(chunk_dir, f"processed_chunk_{k + 1:03d}.xlsx")
            # write-then-rename so an interrupted write never looks finished
            pd.DataFrame({"Question": chunks[k], "Response": replies}).to_excel(
                path + ".tmp.xlsx", index=False)
            os.replace(path + ".tmp.xlsx", path)

    parts = sorted(glob.glob(os.path.join(chunk_dir, "processed_chunk_*.xlsx")))
    full = pd.concat([pd.read_excel(p) for p in parts], ignore_index=True)
    out = os.path.join(DATA_DIR, f"{args.name}.xlsx")
    full.to_excel(out, index=False)
    print(f"wrote {out} ({len(full)} rows from {len(parts)} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
