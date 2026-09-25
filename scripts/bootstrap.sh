#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${repo_root}/var/whisper.cpp"
repository_url="https://github.com/ggml-org/whisper.cpp.git"
revision="${WHISPER_CPP_REVISION:-d09f61a708f3487afa956ff578e60eae5e7a233c}"
model="${WHISPER_MODEL:-medium.en}"

case "${model}" in
  medium.en) expected_model_sha256="cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356" ;;
  small.en) expected_model_sha256="c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d" ;;
  base.en) expected_model_sha256="3e2eabc347eb339c98b417d1eae3c2fc701d0a9ee23c67ca15db49cfa2c16c86" ;;
  *) echo "Pulse Local supports medium.en, small.en, and base.en" >&2; exit 2 ;;
esac

cmake_bin="${CMAKE_BIN:-$(command -v cmake)}"
if [[ -x /opt/homebrew/bin/cmake ]]; then cmake_bin="/opt/homebrew/bin/cmake"; fi
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
[[ -n "${cmake_bin}" ]] || { echo "cmake is required" >&2; exit 1; }

if [[ ! -d "${runtime_root}/.git" ]]; then git clone "${repository_url}" "${runtime_root}"; fi
git -C "${runtime_root}" fetch --depth 1 origin "${revision}"
git -C "${runtime_root}" checkout --detach "${revision}"
arch -arm64 "${cmake_bin}" -S "${runtime_root}" -B "${runtime_root}/build-arm64" -DGGML_METAL=ON -DGGML_NATIVE=OFF -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Release
arch -arm64 "${cmake_bin}" --build "${runtime_root}/build-arm64" --target whisper-server --config Release -j

model_path="${runtime_root}/models/ggml-${model}.bin"
if [[ -f "${model_path}" ]] && [[ "$(shasum -a 256 "${model_path}" | awk '{print $1}')" != "${expected_model_sha256}" ]]; then
  mv "${model_path}" "${model_path}.invalid-$(shasum -a 256 "${model_path}" | awk '{print $1}')"
fi
"${runtime_root}/models/download-ggml-model.sh" "${model}" "${runtime_root}/models"
actual_model_sha256="$(shasum -a 256 "${model_path}" | awk '{print $1}')"
[[ "${actual_model_sha256}" == "${expected_model_sha256}" ]] || { echo "model checksum mismatch" >&2; exit 1; }

python3 - "${runtime_root}" "${repository_url}" "${revision}" "${model}" "${expected_model_sha256}" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
model = root / "models" / f"ggml-{sys.argv[4]}.bin"
manifest = {
    "repository": sys.argv[2],
    "requested_revision": sys.argv[3],
    "resolved_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "model": sys.argv[4],
    "model_sha256": sys.argv[5],
}
(root / "bootstrap-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
