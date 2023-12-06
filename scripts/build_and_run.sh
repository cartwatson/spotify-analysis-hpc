#!/bin/bash

# IMPLEMENTATIONS
serial_implementation() {
    echo "Compiling serial program..."
    if [ -n "$TESTING" ]; then
        g++ -std=c++11 -g -Wall -DTESTING -o serial src/serial.cpp
    else
        g++ -std=c++11 -g -Wall -o serial src/serial.cpp
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed."
        exit 1
    fi

    echo "Running serial program..."
    read -p "Enter command line arguments (enter for none): " -a args
    echo "Program Output:"
    ./serial ${args[@]}

    echo "Program executed successfully"
    echo "Cleaning up..."
    rm serial
}

openmp_implementation() {
    echo "Compiling openmp program..."

    # Detect OS
    OS=$(uname -s)

    # Check if OS is macOS (Apple)
    if [ "$OS" = "Darwin" ]; then
        echo "Detected macOS. Using clang++ for compilation."
        # Check if TESTING environment variable is set
        if [ -n "$TESTING" ]; then
            echo "Compiling with TESTING flag using clang++."
            clang++ -std=c++11 -Xpreprocessor -fopenmp -DTESTING -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp src/omp.cpp -o omp
        else
            echo "Compiling without TESTING flag using clang++."
            clang++ -std=c++11 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp src/omp.cpp -o omp
        fi
    else
        # Check if TESTING environment variable is set
        if [ -n "$TESTING" ]; then
            echo "Compiling with TESTING flag using g++."
            g++ -g -Wall -fopenmp -DTESTING -o omp src/omp.cpp
        else
            echo "Compiling without TESTING flag using g++."
            g++ -g -Wall -fopenmp -o omp src/omp.cpp
        fi
    fi

    # Check for compilation success
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed."
        exit 1
    fi

    echo "Running openmp program..."
    read -p "Enter command line arguments (enter for none): " -a args
    echo "Program Output:"
    ./omp ${args[@]}

    echo "Program executed successfully"
    echo "Cleaning up..."
    rm omp
}

cuda_implementation() {
    echo "Compiling cuda program..."
    if [ -n "$TESTING" ]; then
        nvcc -DTESTING -o cuda src/cuda.cu
    else
        nvcc -o cuda src/cuda.cu
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed."
        exit 1
    fi

    echo "Running cuda program..."
    read -p "Enter command line arguments (enter for none): " -a args
    echo "Program Output:"
    ./cuda ${args[@]}

    echo "Program executed successfully"
    echo "Cleaning up..."
    rm cuda
}

mpi_implementation() {
    echo "Compiling mpi program..."

    mpic++ -std=c++11 -o mpi src/mpi.cpp

    # Check for compilation success
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed."
        exit 1
    fi

    echo "Running mpi program..."
    read -p "Enter command line arguments (enter for none): " -a args
    echo "Program Output:"
    mpirun -np 4 ./mpi ${args[@]}

    echo "Program executed successfully"
    echo "Cleaning up..."
    rm mpi
}

cuda_mpi_implementation() {
    echo "Compiling cuda/mpi program..."
    mpic++ -std=c++11 -c src/mpi_cuda_main.cpp -o mpi_cuda_main.o
    nvcc -c src/mpi_cuda.cu -o mpi_cuda.o
    mpic++ -std=c++11 -o mpi_cuda mpi_cuda_main.o mpi_cuda.o -lmpi -L/usr/local/cuda-12.2/lib64 -lcudart

    echo "Running cuda/mpi program..."
    read -p "Enter command line arguments (enter for none): " -a args
    echo "Program Output:"
    mpirun -n 4 mpi_cuda ${args[@]} 

    echo "Program executed successfully"
    echo "Cleaning up..."
    rm mpi_cuda
    rm mpi_cuda.o
    rm mpi_cuda_main.o
}

# MAIN
echo "Select an implementation to build:"
echo "1. Serial"
echo "2. CPU Shared Memory"
echo "3. GPU Shared Memory"
echo "4. CPU Distributed Memory"
echo "5. GPU Distributed Memory"
read -p "Enter your choice: " choice

case $choice in
    1)
        serial_implementation
        ;;
    2)
        openmp_implementation
        ;;
    3)
        cuda_implementation
        ;;
    4)
        mpi_implementation
        ;;
    5)
        cuda_mpi_implementation
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

exit 0
