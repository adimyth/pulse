#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/validate_fixture.sh /absolute/path/to/video.mp4 [bench options]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$1"
shift

run_gate() {
  local model="$1"
  shift
  WHISPER_MODEL="${model}" bash "${repo_root}/scripts/bootstrap.sh"
  uv run python "${repo_root}/scripts/bench_fixture.py" "${fixture}" --whisper-model "${repo_root}/var/whisper.cpp/models/ggml-${model}.bin" "$@"
}

if run_gate "small.en" "$@"; then
  selected="small.en"
else
  echo "small.en did not pass the local gate; trying base.en" >&2
  run_gate "base.en" "$@"
  selected="base.en"
fi

python3 - "${repo_root}/var/pulse-model" "${selected}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
(root / "stt-selection.json").write_text(json.dumps({"selected_model": sys.argv[2], "selection_rule": "first model passing transcript and timing gates"}, indent=2) + "\n", encoding="utf-8")
PY

echo "Pinned ${selected} for Pulse Local"
