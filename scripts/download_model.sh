#!/bin/bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/download_common.sh"
require_download_tools

download_asset gnn_dyn_model "$REPO_ROOT/data/gnn_dyn_model/2023-01-28-10-42-05-114323/net_epoch_0_iter_1000.pth"
download_asset res_rgr_model "$REPO_ROOT/data/res_rgr_model/2023-01-30-16-17-30-292500/net_best_dy_state_dict.pth"
