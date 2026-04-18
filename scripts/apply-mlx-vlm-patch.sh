#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch_files=(
  "$root_dir/patches/mlx-vlm-streaming-tool-calls.patch"
  "$root_dir/patches/mlx-vlm-reuse-prompt-cache.patch"
)

site_packages="$(
  cd "$root_dir"
  uv run python - <<'PY'
import site

paths = [path for path in site.getsitepackages() if path.endswith("site-packages")]
if not paths:
    raise SystemExit("Could not find site-packages for the project environment.")
print(paths[0])
PY
)"

for patch_file in "${patch_files[@]}"; do
  if (
    cd "$site_packages"
    git apply --reverse --check -p1 "$patch_file"
  ) >/dev/null 2>&1; then
    echo "$(basename "$patch_file") already applied"
    continue
  fi

  (
    cd "$site_packages"
    git apply --check -p1 "$patch_file"
    git apply -p1 "$patch_file"
  )

  echo "Applied $(basename "$patch_file") to $site_packages"
done
