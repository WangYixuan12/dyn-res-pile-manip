#!/bin/bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/download_common.sh"
require_download_tools unzip
PYBULLET_DATA_PATH=$("$DOWNLOAD_PYTHON" -c 'import pybullet_data; print(pybullet_data.__path__[0])')

for robot in kinova franka_panda; do
    archive="$REPO_ROOT/data/$robot.zip"
    download_asset "$robot" "$archive"
    unzip -o "$archive" -d "$PYBULLET_DATA_PATH"
    rm -f -- "$archive"
done
