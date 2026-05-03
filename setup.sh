#!/usr/bin/env bash
set -e

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p dataset/clean dataset/degraded results checkpoints logs

echo "Setup complete. Activate with: source .venv/bin/activate"
