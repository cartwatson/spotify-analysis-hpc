#!/bin/bash

# Ensure that the virtual environment is set up before visualization
if [ ! -d .venv ]; then
    echo "Error: Virtual environment has not been set up."

    read -p "Set up now? [y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Setting up virual environment..."
        python -m venv .venv || python3 -m venv .venv
        echo "Activating venv..."
        if [ -d .venv/Scripts ]; then
            source .venv/Scripts/activate # Windows
        else
            source .venv/bin/activate # Unix
        fi
        echo "Updating pip and installing requirements..."
        python -m pip install --upgrade pip
        pip install -r scripts/requirements.txt
    else
        echo "Exiting."
        exit 1
    fi
fi

echo "Running python visualization..."
if [ -d .venv/Scripts ]; then
    source .venv/Scripts/activate # Windows
else
    source .venv/bin/activate # Unix
fi
python src/python/visualize.py
