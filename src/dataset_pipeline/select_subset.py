"""
Select 1600 rows from each of LUH-data-1 / LUH-data-2 for stage-2 processing.

The two files hold the SAME 3200 questions answered by two DIFFERENT models,
in the same order, so the same row indices are applied to both. That keeps the
design paired: any difference between the two resulting UQ heads is
attributable to the model, not to which questions each one saw. The outputs
stay in separate files -- they are never mixed.

Why not simply take the first half: the file is ordered alphabetically by the
underlying English medical term, and is heterogeneous -- roughly idx 0-1000 is
anatomy/lab-tests plus synthetic "نوع N" variants, and idx ~1000-3199 is a real
A-Z disease list. head(1600) would therefore yield every synthetic row plus
diseases only from A-C, and nothing from D-Z.

Selection policy (all deterministic -- no RNG, so it is reproducible):
  1. Group rows by base entity (strip "نوع N", take the first "|" alias,
     normalise ZWNJ/whitespace) and keep at most ONE row per base entity, so
     budget is not spent on near-duplicates (e.g. بیهوشی occurs 8 times).
  2. Cap synthetic "نوع N" rows at ~6% of the selection (they are mostly
     non-existent entities such as "Axon Type 2" that the models confabulate
     about; useful as hallucination positives, but 19.5% of the set would teach
     the head to spot the artifact rather than genuine hallucination).
     Preference is given to the 59 bases that exist ONLY as a synthetic
     variant, so no real entity is displaced.
  3. Sample the remainder systematically (evenly spaced) across the
     alphabetically ordered pool, preserving A-Z topic coverage.

Run:
  python select_subset.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

from paths import DATA_DIR

DATASETS = ["LUH-data-1", "LUH-data-2"]
TARGET_ROWS = 1600
SYNTH_FRACTION = 0.06  # ~6%, inside the agreed 5-8% band

ENTITY_RE = re.compile(r"در مورد\s+(.+?)\s+هر ?جه میدانید")
SYNTH_RE = re.compile(r"نوع\s*\d")
ZWNJ = "‌"


def extract_entity(question):
    m = ENTITY_RE.search(str(question))
    return m.group(1) if m else ""


def base_entity(entity):
    """Collapse 'X نوع 3' and 'X | alias' down to a comparable base name."""
    s = entity.replace(ZWNJ, " ")
    s = SYNTH_RE.sub("", s)
    s = s.replace("نوع", "")
    s = s.split("|")[0]
    return re.sub(r"\s+", " ", s).strip()


def evenly_spaced(items, k):
    """k evenly spaced elements of `items`, preserving order. Systematic
    sampling: keeps coverage across the whole (alphabetical) range instead of
    clustering, and is deterministic."""
    if k >= len(items):
        return list(items)
    if k <= 0:
        return []
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    idx = sorted(set(idx.tolist()))
    # rounding collisions can leave us short; backfill with unused positions
    if len(idx) < k:
        used = set(idx)
        for j in range(len(items)):
            if len(idx) == k:
                break
            if j not in used:
                idx.append(j)
                used.add(j)
        idx = sorted(idx)
    return [items[j] for j in idx]


def main():
    frames = {}
    for name in DATASETS:
        frames[name] = pd.read_excel(os.path.join(DATA_DIR, f"{name}.xlsx"))

    # The whole paired design rests on this; verify rather than assume.
    q0 = frames[DATASETS[0]]["Question"].tolist()
    for name in DATASETS[1:]:
        if frames[name]["Question"].tolist() != q0:
            print(f"ABORT: questions in {name} differ from {DATASETS[0]}; "
                  f"shared indices would not be a paired design.")
            return 1
    print(f"Verified: all {len(DATASETS)} datasets share identical questions in "
          f"the same order ({len(q0)} rows).")

    df = frames[DATASETS[0]]
    ent = df["Question"].map(extract_entity)
    base = ent.map(base_entity)
    is_synth = ent.str.contains(SYNTH_RE)

    # ---- group by base entity, pick one representative row per base --------
    real_rep, synth_rep = {}, {}
    for i in range(len(df)):
        b = base.iloc[i]
        if not b:
            continue
        if is_synth.iloc[i]:
            synth_rep.setdefault(b, i)
        else:
            real_rep.setdefault(b, i)

    synth_only = sorted(b for b in synth_rep if b not in real_rep)
    synth_dual = sorted(b for b in synth_rep if b in real_rep)

    print(f"\nBase entities: {len(set(base) - {''})} unique")
    print(f"  with a real row      : {len(real_rep)}")
    print(f"  with a synthetic row : {len(synth_rep)}  "
          f"(of which {len(synth_only)} are synthetic-only)")

    # ---- synthetic quota ---------------------------------------------------
    n_synth = round(TARGET_ROWS * SYNTH_FRACTION)
    chosen_synth_bases = list(synth_only[:n_synth])
    if len(chosen_synth_bases) < n_synth:
        need = n_synth - len(chosen_synth_bases)
        chosen_synth_bases += evenly_spaced(synth_dual, need)
    chosen_synth_bases = set(chosen_synth_bases)
    synth_idx = sorted(synth_rep[b] for b in chosen_synth_bases)

    # ---- real remainder, systematically spaced over the alphabetical pool --
    n_real = TARGET_ROWS - len(synth_idx)
    real_pool = sorted(i for b, i in real_rep.items() if b not in chosen_synth_bases)
    if n_real > len(real_pool):
        print(f"ABORT: need {n_real} real rows but pool has only {len(real_pool)}.")
        return 1
    real_idx = evenly_spaced(real_pool, n_real)

    selected = sorted(set(real_idx) | set(synth_idx))

    # ---- verify the policy actually held ----------------------------------
    print("\n" + "-" * 74)
    print("VERIFICATION")
    print("-" * 74)
    sel_base = base.iloc[selected]
    sel_synth = is_synth.iloc[selected]
    checks = [
        ("exactly TARGET_ROWS selected", len(selected) == TARGET_ROWS,
         f"{len(selected)} / {TARGET_ROWS}"),
        ("no duplicate row indices", len(selected) == len(set(selected)), ""),
        ("<=1 row per base entity", sel_base.nunique() == len(selected),
         f"{sel_base.nunique()} distinct bases / {len(selected)} rows"),
        ("no duplicate question text", df["Question"].iloc[selected].nunique() == len(selected),
         f"{df['Question'].iloc[selected].nunique()} distinct questions"),
        ("synthetic share within 5-8%", 0.05 <= sel_synth.mean() <= 0.08,
         f"{100 * sel_synth.mean():.1f}%  ({sel_synth.sum()} rows)"),
    ]
    ok = True
    for label, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail else ""))

    print("\n  A-Z coverage (selected rows per original 400-row block):")
    arr = np.array(selected)
    for s in range(0, len(df), 400):
        n = int(((arr >= s) & (arr < s + 400)).sum())
        bar = "#" * round(n / 8)
        print(f"    idx {s:4}-{s + 399:<4} {n:4}  {bar}")

    if not ok:
        print("\nVerification failed; not writing output.")
        return 1

    # ---- write one output per dataset, never merged ------------------------
    print("\n" + "-" * 74)
    print("OUTPUT")
    print("-" * 74)
    for name in DATASETS:
        sub = frames[name].iloc[selected].reset_index(drop=True)
        out = os.path.join(DATA_DIR, f"{name}-selected.xlsx")
        sub.to_excel(out, index=False)
        chars = sub["Response"].astype(str).str.len()
        print(f"  {out}")
        print(f"      rows={len(sub)}  reply chars mean={chars.mean():.0f} "
              f"median={chars.median():.0f}")

    idx_path = os.path.join(DATA_DIR, "selected_indices.csv")
    pd.DataFrame(
        {
            "row_index": selected,
            "entity": ent.iloc[selected].values,
            "base_entity": sel_base.values,
            "is_synthetic": sel_synth.values,
        }
    ).to_csv(idx_path, index=False)
    print(f"  {idx_path}   (shared index list -- same rows in both datasets)")

    print("\nBoth files use the SAME row indices, so the two sets stay paired "
          "for model comparison, and remain in separate files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
