#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
patch_file="$root_dir/patches/mlx-vlm-streaming-tool-calls.patch"

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

if (
  cd "$site_packages"
  git apply --reverse --check -p1 "$patch_file"
) >/dev/null 2>&1; then
  echo "mlx-vlm patch already applied"
  exit 0
fi

(
  cd "$site_packages"
  git apply --check -p1 "$patch_file"
  git apply -p1 "$patch_file"
)

echo "Applied mlx-vlm patch to $site_packages"
