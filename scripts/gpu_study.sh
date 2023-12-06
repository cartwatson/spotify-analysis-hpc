#!/bin/bash

# Define the block sizes to test
BLOCK_SIZES=(32 64 128 256)

# Compile the CUDA program
echo "Compiling CUDA program..."
nvcc -o cuda src/cuda.cu
if [ $? -ne 0 ]; then
    echo "Error: Compilation of CUDA program failed."
    exit 1
fi
echo "CUDA program compilation successful."

# Compile the CUDA/MPI program
echo "Compiling CUDA/MPI program..."
mpic++ -std=c++11 -c src/mpi_cuda_main.cpp -o mpi_cuda_main.o
nvcc -c src/mpi_cuda.cu -o mpi_cuda.o
mpic++ -std=c++11 -o mpi_cuda mpi_cuda_main.o mpi_cuda.o -lmpi -L/usr/local/cuda-12.2/lib64 -lcudart
if [ $? -ne 0 ]; then
    echo "Error: Compilation of CUDA/MPI program failed."
    exit 1
fi
echo "CUDA/MPI program compilation successful."

# Directory for logs
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Files to store results
CUDA_RESULT_FILE="${LOG_DIR}/cuda_scaling_results.csv"
CUDA_MPI_RESULT_FILE="${LOG_DIR}/cuda_mpi_scaling_results.csv"
echo "Block Size,K-means Execution Time" > $CUDA_RESULT_FILE
echo "Block Size,K-means Execution Time" > $CUDA_MPI_RESULT_FILE

# Function to extract time from the program output
extract_time() {
    echo $(echo "$1" | grep "$2" | grep -o -E '[0-9]+\.[0-9]+')
}

# Function to run a program and log results
run_and_log() {
    echo "Running $1 program with block size: $2"
    output=$($3 $2 2>&1)
    k_means_time=$(extract_time "$output" "Finished k-means in")
    echo "$2,$k_means_time" >> $4
}

# Iterate over each block size and run tests for both CUDA and CUDA/MPI
for block_size in "${BLOCK_SIZES[@]}"; do
    run_and_log "CUDA" $block_size "./cuda 250000" $CUDA_RESULT_FILE
    run_and_log "CUDA/MPI" $block_size "mpirun -n 4 ./mpi_cuda 250000" $CUDA_MPI_RESULT_FILE
done

echo "GPU scaling study completed. Results saved in $LOG_DIR"

# Cleaning up
echo "Cleaning up..."
rm cuda mpi_cuda mpi_cuda.o mpi_cuda_main.o
