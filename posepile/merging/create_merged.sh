#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

python -m posepile.merging.merged_dataset3d --name=huge8
