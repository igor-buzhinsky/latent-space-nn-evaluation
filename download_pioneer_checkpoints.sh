#!/usr/bin/env bash
# Downloads the pretrained PIONEER checkpoints (~670 MB each) needed for the CelebA and LSUN experiments.
# Checkpoints that already exist are skipped. Files are hosted on Google Drive and downloaded with gdown, which is
# installed into the project environment by poetry install.
# Usage: ./download_pioneer_checkpoints.sh

set -euo pipefail
cd "$(dirname "$0")"

# destination path and Google Drive file ID of each checkpoint
checkpoints=(
    "pioneer/CelebA-128-female/checkpoint/20580000_state 1X1nkyK3hkaahBYRfH36X5yyInZGYrQbW"
    "pioneer/CelebA-128-male/checkpoint/22452000_state 1hWpm1vLXd_ay2M4AxOMxLjw4mD9gyZzH"
    "pioneer/LSUN-128-bedroom/checkpoint/15268000_state 1sz-_3SsENJ9a4OVUF2o6q7riCcgeVlWS"
    "pioneer/LSUN-128-church_outdoor/checkpoint/20212000_state 1eqGKeHQf-KvAe7zrvFiYjqqrEwsxvcwb"
)

for entry in "${checkpoints[@]}"; do
    read -r path file_id <<< "$entry"
    if [[ -f "$path" ]]; then
        echo "Skipping $path: already exists."
        continue
    fi
    mkdir -p "$(dirname "$path")"
    # download to a temporary name so that an interrupted download is not mistaken for a complete checkpoint
    poetry run gdown "https://drive.google.com/uc?id=$file_id" -O "$path.part"
    mv "$path.part" "$path"
done
