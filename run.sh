#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate $1
python gpu-occ.py
