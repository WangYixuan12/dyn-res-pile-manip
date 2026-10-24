#!/bin/bash

# Shared by the release download scripts; source this file with Bash.
DOWNLOAD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$DOWNLOAD_SCRIPT_DIR/.." && pwd)

require_download_tools() {
    local tool
    for tool in gdown "$@"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "Required command not found: $tool. Activate the project environment and install the missing dependency." >&2
            return 1
        fi
    done
    if command -v python >/dev/null 2>&1; then
        DOWNLOAD_PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        DOWNLOAD_PYTHON=python3
    else
        echo "Python is required to verify download checksums." >&2
        return 1
    fi
}

verify_download() {
    "$DOWNLOAD_PYTHON" - "$1" "$2" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != sys.argv[2]:
    print("SHA-256 mismatch for {}: expected {}, got {}".format(
        sys.argv[1], sys.argv[2], actual), file=sys.stderr)
    sys.exit(1)
PY
}

# Use a subshell so temporary-file cleanup does not replace the caller's traps.
download_asset() (
    local resource=$1 destination=$2 key file_id sha256
    local selected_id="" selected_sha="" temporary=""
    while IFS=$'\t' read -r key file_id sha256; do
        if [[ "$key" == "$resource" ]]; then
            selected_id=$file_id
            selected_sha=$sha256
            break
        fi
    done < "$DOWNLOAD_SCRIPT_DIR/download_assets.tsv"
    if [[ -z "$selected_id" || "$selected_id" == TODO_* || ! "$selected_sha" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Missing release metadata for $resource in scripts/download_assets.tsv." >&2
        exit 1
    fi

    mkdir -p -- "$(dirname -- "$destination")"
    if [[ -f "$destination" ]] && verify_download "$destination" "$selected_sha"; then
        echo "Using verified file: $destination"
        exit 0
    fi

    temporary=$(mktemp "${destination}.part.XXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    gdown "https://drive.google.com/uc?id=$selected_id" --output "$temporary"
    verify_download "$temporary" "$selected_sha"
    mv -f -- "$temporary" "$destination"
)
