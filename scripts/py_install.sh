#!/bin/bash

echo "Setting up virual environment..."
python -m venv .venv

echo "Activating venv..."
source .venv/Scripts/activate || source .venv/bin/activate

echo "Updating pip and installing requirements..."
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
