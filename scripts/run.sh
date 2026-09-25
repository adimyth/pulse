#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
selection_file="${repo_root}/var/pulse-model/stt-selection.json"
selection_args=()
if [[ -f "${selection_file}" ]]; then
  model="$(python3 - "${selection_file}" <<'PY'
import json
import pathlib
import sys

model = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")).get("selected_model")
if model not in {"small", "base", "small.en", "base.en"}:
    raise SystemExit("stt selection is invalid")
print(model)
PY
)"
  selection_args=(--whisper-model "${repo_root}/var/whisper.cpp/models/ggml-${model}.bin")
fi
uv run python -m pulse.server "${selection_args[@]}" "$@"
