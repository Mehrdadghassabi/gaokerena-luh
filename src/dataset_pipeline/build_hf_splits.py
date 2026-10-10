"""
Build a DatasetDict with train/eval/test splits from the finalized stage-2
dataset, and optionally push it to the Hugging Face Hub.

Split sizes: test=50, eval=92, train=rest (matching the layout requested).

The rows are ordered alphabetically by the underlying English medical term, so
taking contiguous slices would give test/eval only entities from one part of the
alphabet. Instead each split is drawn systematically (evenly spaced) across the
whole range, so all three cover A-Z. Deterministic -- no RNG.

Build locally only (default):
    python build_hf_splits.py --dataset LUH-data-1-selected

Build and push (explicit opt-in):
    python build_hf_splits.py --dataset LUH-data-1-selected --push \
        --repo gaokerena/LUH_Gaokerena_V [--private]
"""
import argparse
import math
import os
import sys

import numpy as np

from datasets import DatasetDict, load_from_disk

from paths import WORK_DIR

N_TEST = 50
N_EVAL = 92


def evenly_spaced(pool, k):
    """k evenly spaced elements of `pool`, order preserved (systematic sampling)."""
    if k >= len(pool):
        return list(pool)
    idx = sorted(set(np.linspace(0, len(pool) - 1, k).round().astype(int).tolist()))
    if len(idx) < k:  # rounding collisions -> backfill
        used = set(idx)
        for j in range(len(pool)):
            if len(idx) == k:
                break
            if j not in used:
                idx.append(j)
                used.add(j)
        idx = sorted(idx)
    return [pool[j] for j in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="LUH-data-1-selected",
                    help="stage-2 run to split (default: %(default)s)")
    ap.add_argument("--src", default=None,
                    help="default: <LUH_WORK_DIR>/stage2/<dataset>/hf_dataset")
    ap.add_argument("--out", default=None, help="default: <src>/../hf_dataset_splits")
    ap.add_argument("--push", action="store_true", help="push to the Hub")
    ap.add_argument("--repo", default=None, help="e.g. gaokerena/LUH_Gaokerena_V")
    ap.add_argument("--private", action="store_true",
                    help="create as private (only affects repos that do not exist yet)")
    args = ap.parse_args()

    args.src = args.src or os.path.join(WORK_DIR, "stage2", args.dataset, "hf_dataset")
    out_dir = args.out or os.path.join(os.path.dirname(args.src), "hf_dataset_splits")

    ds = load_from_disk(args.src)
    n = len(ds)
    print(f"source: {args.src}  ({n} rows)")

    if N_TEST + N_EVAL > n:
        print(f"ABORT: need {N_TEST + N_EVAL} rows for test+eval but only {n} exist.")
        return 1

    all_idx = list(range(n))
    test_idx = evenly_spaced(all_idx, N_TEST)
    rest = [i for i in all_idx if i not in set(test_idx)]
    eval_idx = evenly_spaced(rest, N_EVAL)
    train_idx = [i for i in rest if i not in set(eval_idx)]

    dd = DatasetDict({
        "train": ds.select(train_idx),
        "eval": ds.select(eval_idx),
        "test": ds.select(test_idx),
    })

    # ---------------------------------------------------------- verification
    print("\n" + "-" * 70)
    print("VERIFICATION")
    print("-" * 70)
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    check("split sizes as requested",
          len(dd["test"]) == N_TEST and len(dd["eval"]) == N_EVAL
          and len(dd["train"]) == n - N_TEST - N_EVAL,
          f"train={len(dd['train'])} eval={len(dd['eval'])} test={len(dd['test'])}")
    check("splits are disjoint",
          not (set(train_idx) & set(eval_idx)) and not (set(train_idx) & set(test_idx))
          and not (set(eval_idx) & set(test_idx)))
    check("splits cover every row",
          len(set(train_idx) | set(eval_idx) | set(test_idx)) == n)
    check("no duplicate questions across splits",
          len({q for s in dd for q in dd[s]["question"]}) == n)

    # A-Z coverage: each split should span the whole index range, not cluster
    for s, idxs in (("train", train_idx), ("eval", eval_idx), ("test", test_idx)):
        spread = (min(idxs), max(idxs))
        blocks = len({i // 200 for i in idxs})
        check(f"{s} covers the full alphabetical range",
              spread[0] < n * 0.05 and spread[1] > n * 0.95 and blocks == math.ceil(n / 200),
              f"idx {spread[0]}..{spread[1]}, {blocks}/{math.ceil(n / 200)} blocks touched")

    for s in dd:
        claims = sum(len(c) for c in dd[s]["claims"])
        allv = [v for row in dd[s]["verified"] for v in row]
        valid = [v for v in allv if v is not None and not np.isnan(v)]
        halluc = 100 * sum(1 for v in valid if v == 1) / max(len(valid), 1)
        check(f"{s} has both classes in 'verified'", len(set(valid)) == 2,
              f"{len(dd[s])} rows, {claims} claims, {halluc:.1f}% hallucinated")

    if not ok:
        print("\nVerification failed -- not writing or pushing.")
        return 1

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    dd.save_to_disk(out_dir)
    print(f"\nsaved locally -> {out_dir}")
    print(f"features: {dd['train'].features}")

    if args.push:
        if not args.repo:
            print("ABORT: --push requires --repo")
            return 1
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("ABORT: HF_TOKEN not set")
            return 1
        print(f"\npushing to https://huggingface.co/datasets/{args.repo} ...")
        dd.push_to_hub(repo_id=args.repo, token=token, private=args.private)
        print(f"done -> https://huggingface.co/datasets/{args.repo}")
    else:
        print("\n(not pushed -- pass --push --repo <org/name> to upload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
