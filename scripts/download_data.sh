#!/bin/bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/download_common.sh"

usage() {
    echo "Usage: bash scripts/download_data.sh [gnn_dyn_data|res_rgr_data|res_rgr_data_small ...|all]"
    echo "With no arguments, downloads gnn_dyn_data only."
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# == 0 ]]; then
    set -- gnn_dyn_data
elif [[ $# == 1 && $1 == all ]]; then
    set -- gnn_dyn_data res_rgr_data res_rgr_data_small
fi
for dataset in "$@"; do
    case "$dataset" in
        gnn_dyn_data|res_rgr_data|res_rgr_data_small) ;;
        *) usage >&2; exit 1 ;;
    esac
done

require_download_tools unzip
for dataset in "$@"; do
    archive="$REPO_ROOT/data/$dataset.zip"
    download_asset "$dataset" "$archive"
    unzip -o "$archive" -d "$REPO_ROOT/data/$dataset"
    rm -f -- "$archive"
done
