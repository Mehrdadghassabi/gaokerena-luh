# LUH dataset pipeline

Builds the token-level hallucination datasets (`LUH-data-1`, `LUH-data-2`) used to train the uncertainty head. The folder is self-contained: it does not import the `luh` package or anything else from the parent repo, and it can be copied anywhere.

| Stage | Script | Input → output |
|---|---|---|
| 1. Answers | `generate_answers.py` | questions → `data/<name>/processed_chunk_NNN.xlsx` + `data/<name>.xlsx` |
| 1.5. Subset | `select_subset.py` | `data/LUH-data-{1,2}.xlsx` → `data/LUH-data-{1,2}-selected.xlsx` + `data/selected_indices.csv` |
| 2. Claims + fact check | `run_stage2.py` | `data/<dataset>.xlsx` → `workdir/stage2/<dataset>/` (`rows.jsonl`, then `hf_dataset`) |
| 3. Splits | `build_hf_splits.py` | `hf_dataset` → `hf_dataset_splits` (train/eval/test = rest/92/50), optional Hub push |

`data/` ships with `LUH-data-1.xlsx` and `LUH-data-2.xlsx`, the stage 1 outputs. They hold the same 3,200 questions in the same order, answered by two different models (data-2 by `gaokerena/gaokerena-r1.0`). Stage 1.5 applies one shared set of row indices to both, so the two datasets stay paired.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # then fill in the key(s)
huggingface-cli login   # aya-expanse-8b (the tokenizer for stage 2) is gated
```

`.env` (git-ignored) is read automatically and does not override variables already exported:

```
OPENROUTER_API_KEY=sk-or-...
DEEPSEEK_API_KEY=sk-...        # only for --provider deepseek
```

Optional: `LUH_DATA_DIR` and `LUH_WORK_DIR` move the data and run directories (defaults: `./data` and `./workdir`).

## Running

```bash
export PYTHON=.venv/bin/python
./run.sh answers LUH-data-1 CohereLabs/aya-expanse-8b                  # stage 1, local GPU
./run.sh answers LUH-data-3 cohere/command-r7b-12-2024 --backend openrouter
./run.sh select                                                        # stage 1.5
./run.sh stage2 LUH-data-1-selected          # stage 2: all 1600 rows, resumable
./run.sh status LUH-data-1-selected
./run.sh finalize LUH-data-1-selected        # hf_dataset + splits
```

Every stage can be resumed: re-run the same command after any interruption. Stage 2 caches every API response on disk, so a resume costs nothing extra. `--cache-dir` points stage 2 at an earlier run's cache to replay it.

## Things to know

- **Stage 1 is reconstructed.** The script that produced the existing answers was never saved (it was likely run in Colab). `generate_answers.py` is rebuilt from the upstream `run_generate_texts.py` plain-HF path, using greedy decoding. It uses `max_new_tokens=1024`: the existing replies run to a median of about 700 tokens and cap at exactly 1,024, so the 256 in the original spec is wrong. Its output should be close to the existing files but is not guaranteed identical.
- **OpenRouter does not host `aya-expanse-8b` or `gaokerena-r1.0`.** `--backend openrouter` is for answering with other models; it cannot regenerate data-1/-2.
- **Where the questions came from is unknown.** The 3,200 questions don't appear in `gaokerena/fa_luh_aya`. The only copy is the `Question` column of `data/LUH-data-1.xlsx`, which stage 1 reads by default.
- **Keep one judge per dataset.** The `openrouter` provider uses the pinned `deepseek/deepseek-v4-flash-0731`, which labelled data-1. The official DeepSeek API only has the rolling `deepseek-v4-flash`, which is a different judge. Each run records which judge labelled which rows in `judge.jsonl`.
- **Cost estimates in `run_stage2.py` use the prices from the data-1 run** ($0.09/M in, $0.18/M out). OpenRouter now lists `-0731` at $0.0168/M in and $1.28/M out (checked 2026-10-10), so new runs will cost more than the printed estimate.
