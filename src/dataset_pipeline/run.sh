#!/usr/bin/env bash
# Convenience wrapper for the LUH dataset pipeline. Run from anywhere.
#
#   ./run.sh answers LUH-data-1 CohereLabs/aya-expanse-8b    # stage 1 (GPU)
#   ./run.sh answers LUH-data-2 gaokerena/gaokerena-r1.0
#   ./run.sh select                                          # stage 1.5
#   ./run.sh stage2 LUH-data-1-selected [rows] [provider]    # stage 2, resumable
#   ./run.sh status LUH-data-1-selected
#   ./run.sh finalize LUH-data-1-selected                    # stage 2 -> HF dataset -> splits
#
# stage2 defaults: all 1600 rows, OpenRouter, pinned deepseek/deepseek-v4-flash-0731
# (the judge that labelled data-1). provider=deepseek uses the official API, whose
# rolling deepseek-v4-flash is a DIFFERENT judge -- don't mix the two in one dataset.
#
# Set PYTHON to pick an interpreter (default: python3).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
WORK="${LUH_WORK_DIR:-workdir}"

case "${1:-}" in
  answers)
    exec "$PY" generate_answers.py --name "$2" --model "$3" "${@:4}"
    ;;
  select)
    exec "$PY" select_subset.py
    ;;
  stage2)
    : "${HF_TOKEN:=$(cat ~/.cache/huggingface/token 2>/dev/null)}"
    export HF_TOKEN
    ROWS="${3:-1600}"
    case "${4:-openrouter}" in
      openrouter)
        exec "$PY" run_stage2.py --provider openrouter \
          --model deepseek/deepseek-v4-flash-0731 \
          --dataset "$2" --rows "$ROWS" --threads 16 ;;
      deepseek)
        exec "$PY" run_stage2.py --provider deepseek \
          --dataset "$2" --rows "$ROWS" --threads 16 ;;
      *) echo "provider must be openrouter or deepseek" >&2; exit 2 ;;
    esac
    ;;
  status)
    OUT="${WORK}/stage2/$2"
    echo "rows done: $(wc -l < "${OUT}/rows.jsonl" 2>/dev/null || echo 0)"
    [ -f "${OUT}/progress.json" ] && cat "${OUT}/progress.json"
    echo
    ;;
  finalize)
    "$PY" run_stage2.py --dataset "$2" --finalize || exit 1
    exec "$PY" build_hf_splits.py --dataset "$2"
    ;;
  *)
    sed -n '2,15p' "$0"; exit 2
    ;;
esac
