#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Für nicht offiziell unterstützte Karten ggf. setzen, z.B. gfx1030 für RDNA2:
# export HSA_OVERRIDE_GFX_VERSION=10.3.0

exec uv run --extra rocm main.py --disable-smart-memory "$@"
