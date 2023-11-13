#!/bin/bash

# Set up virtual environment
setup_venv() {
	echo "Setting up virtual environment..."
	if command -v python &> /dev/null; then
		python -m venv .venv || { echo "Error: Failed to create virtual environment. Make sure 'python' is installed."; exit 1; }
	elif command -v python3 &> /dev/null; then
		python3 -m venv .venv || { echo "Error: Failed to create virtual environment. Make sure 'python3' is installed."; exit 1; }
	else
		echo "Error: Neither 'python' nor 'python3' found. Please install Python before proceeding."
		exit 1
	fi

	echo "Activating venv..."
	if [[ -n "$WINDIR" ]]; then # Windows
		source .venv/Scripts/activate
	else # UNIX
		source .venv/bin/activate
	fi

	echo "Updating pip and installing requirements..."
	python -m pip install --upgrade pip || echo "Error: Failed to upgrade pip."
	pip install -r scripts/requirements.txt || { echo "Error: Failed to install requirements."; exit 1; }

	deactivate
}

# Run Python visualization
run_visualization() {
	echo "Running python visualization...";
	if [[ -n "$WINDIR" ]]; then # Windows
		source .venv/Scripts/activate
	else # UNIX
		source .venv/bin/activate
	fi
	python ./src/python/visualize.py || echo "Error: Failed to run visualization script."
	deactivate
}

# Ensure that the virtual environment is set up before visualization
if [ ! -d .venv ]; then
	echo "A virtual environment for this script has not been set up."

	read -p "Set up now? [y/n] " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		setup_venv
	else
		echo "Exiting."
		exit 1
	fi
fi

# Run the visualization
run_visualization
