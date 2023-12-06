#!/bin/bash

# Define the sizes of datasets to test
DATASET_SIZES=(50000 100000 150000 200000)

# Compile the serial, OpenMP, and MPI programs
compile_program() {
    echo "Compiling $1 program..."
    $2
    if [ $? -ne 0 ]; then
        echo "Error: Compilation of $1 failed."
        exit 1
    fi
    echo "Compilation of $1 successful."
}

# Compile serial
compile_program "serial" "g++ -std=c++11 -g -Wall -DTESTING -o serial src/serial.cpp"

# Compile OpenMP
OS=$(uname -s)
if [ "$OS" = "Darwin" ]; then
    compile_command="clang++ -std=c++11 -Xpreprocessor -fopenmp -DTESTING -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp src/omp.cpp -o omp"
else
    compile_command="g++ -g -Wall -fopenmp -DTESTING -o omp src/omp.cpp"
fi
compile_program "OpenMP" "$compile_command"

# Compile MPI
compile_program "MPI" "mpic++ -std=c++11 -o mpi -DTESTING src/mpi.cpp"

# Directory for logs
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Files to store results
SERIAL_RESULT_FILE="${LOG_DIR}/serial_scaling_results.csv"
OMP_RESULT_FILE="${LOG_DIR}/omp_scaling_results.csv"
MPI_RESULT_FILE="${LOG_DIR}/mpi_scaling_results.csv"
echo "Dataset Size,Data Parsing Time,K-means Execution Time" > $SERIAL_RESULT_FILE
echo "Dataset Size,Data Parsing Time,K-means Execution Time" > $OMP_RESULT_FILE
echo "Dataset Size,Data Parsing Time,K-means Execution Time" > $MPI_RESULT_FILE

# Function to extract time from the program output
extract_time() {
    echo $(echo "$1" | grep "$2" | grep -o -E '[0-9]+\.[0-9]+')
}

# Function to run a program and log results
run_and_log() {
    echo "Running $1 for dataset size: $2"
    output=$($3 $2 2>&1)
    parse_time=$(extract_time "$output" "Parsed data in")
    k_means_time=$(extract_time "$output" "Finished k-means in")
    echo "$2,$parse_time,$k_means_time" >> $4
}

# Iterate over each dataset size and run tests
for size in "${DATASET_SIZES[@]}"; do
    run_and_log "serial" $size "./serial" $SERIAL_RESULT_FILE
    run_and_log "OpenMP" $size "./omp" $OMP_RESULT_FILE
    run_and_log "MPI" $size "mpirun -np 4 ./mpi" $MPI_RESULT_FILE
done

echo "Scaling study completed. Results saved in $LOG_DIR"

# Cleaning up
echo "Cleaning up..."
rm serial omp mpi
