## Build:
`g++ -g -Wall -o serial serial.cpp`

## Execute:
`./serial [numLines]`
- `numLines` is the number of lines to read from `tracks_features.csv` (useful for quick debugging)
- Keep `numLines` empty to read all lines 

## Installing dependencies:
`./py_install.sh`
- Sets up virtual environment and installs all dependencies required in order to run `visualize.py`

## Visualization:
`python visualize.py`
- **Must be in virtual environment with libraries installed**
    - Activate virtual environment with `source venv/bin/activate` or `source venv/Scripts/activate` (Windows)
    - Deactivate virtual environment with `deactivate`
