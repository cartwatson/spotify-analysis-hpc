#!/bin/bash

# catch errors
trap 'echo "An error occurred. Exiting..."; deactivate; exit 1;' ERR
set -e

echo "Setting up virual environment..."
python -m venv .venv || python3 -m venv .venv

echo "Activating venv..."
source .venv/Scripts/activate || source .venv/bin/activate

echo "Updating pip and installing requirements..."
python -m pip install --upgrade pip || python3 -m pip install --upgrade pip
pip install -r scripts/requirements.txt
deactivate
