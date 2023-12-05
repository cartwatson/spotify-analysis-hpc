# Build and Execution Instructions

*`scripts/build_and_run.sh` provides a command line interface to build and run all implementations.*

## 1. Serial Implementation

- **Build:** `g++ -g -Wall -o serial serial.cpp`
- **Execute:** `./serial [numLines]`
  - `numLines` is the number of lines to read from `tracks_features.csv` (useful for quick debugging)
  - Keep `numLines` empty to read all lines 

## 2. Parallel Shared Memory CPU - OpenMp

- **Build:** `g++ -g -Wall -fopenmp -o omp src/omp.cpp`
- **Execute:** `./omp [numLines]`

## 3. Parallel Shared Memory GPU - Cuda

- **Build:** ``
- **Execute:** ``

## 4. Distributed Memory CPU - MPI

- **Build:** ``
- **Execute:** ``

## 5. Distributed Memory GPU - Cuda & MPI

- **Build:** ``
- **Execute:** ``
