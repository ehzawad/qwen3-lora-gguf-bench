#!/usr/bin/env bash
# Fail if any tracked file is a model/binary artifact, is oversized, or lives
# under a runtime-only directory. Run before the first commit and in CI.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fail=0

# Forbidden extensions among tracked files.
if git ls-files | grep -iE '\.(safetensors|gguf|bin|pt|pth|ckpt|onnx|tflite|so|a|o|dylib|dll|exe)$'; then
  echo "hygiene: ERROR - tracked model/binary artifact(s) above" >&2; fail=1
fi

# Forbidden runtime dirs.
if git ls-files | grep -E '^(models|artifacts|vendor|build|logs|run)/'; then
  echo "hygiene: ERROR - tracked file under a runtime-only directory" >&2; fail=1
fi

# Oversized tracked files (>10 MiB).
while read -r f; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt $((10*1024*1024)) ]; then
    echo "hygiene: ERROR - tracked file >10MiB: $f ($sz bytes)" >&2; fail=1
  fi
done < <(git ls-files)

# Gitlinks (submodule leftovers).
if git ls-files -s | awk '$1==160000{print $4}' | grep .; then
  echo "hygiene: ERROR - gitlink (mode 160000) present" >&2; fail=1
fi

if [ "$fail" -eq 0 ]; then echo "hygiene: OK - no artifacts/oversize/gitlinks tracked"; fi
exit "$fail"
