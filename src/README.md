# Compile and Execution Instructions

*[scripts/build_and_run.sh](/scripts/build_and_run.sh) provides a simple command line interface to compile and run all implementations.*

## 1. Serial Implementation

- **Compile:** `g++ -g -Wall -o serial serial.cpp`
- **Execute:** `./serial [numLines]`

## 2. Parallel Shared Memory CPU - OpenMp

- **Compile:** `g++ -g -Wall -fopenmp -o omp src/omp.cpp`
- **Execute:** `./omp [numLines]`
- **NOTE:** This implementation will not run on the CHPC due to differing versions of OpenMP

## 3. Parallel Shared Memory GPU - Cuda

- **Compile:** `nvcc -o cuda src/cuda.cu`
- **Execute:** `./cuda [numLines]`

## 4. Distributed Memory CPU - MPI

- **Compile:** `mpic++ -o mpi src/mpi.cpp`
- **Execute:** `mpirun -np 4 ./mpi [numLines]`

## 5. Distributed Memory GPU - Cuda & MPI

- **Compile:**
```bash
    mpic++ -std=c++11 -c src/mpi_cuda_main.cpp -o mpi_cuda_main.o
    nvcc -c src/mpi_cuda.cu -o mpi_cuda.o
    mpic++ -std=c++11 -o mpi_cuda mpi_cuda_main.o mpi_cuda.o -lmpi -L/usr/local/cuda-12.2/lib64 -lcudart
```
- **Execute:** `mpirun -n 4 ./mpi_cuda [numLines]`

## Notes

- in order to run openmpi on CHPC you will need to run `module load openmpi`
- in order to run cuda on CHPC you will need to run `module load nvhpc`
- all builds can be executed with the argument `[numLines]` which is the number of lines to read from `tracks_features.csv`
  - Keep `numLines` empty to read all lines 
  - This is primarily used for quick debugging
