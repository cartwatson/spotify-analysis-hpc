#!/bin/bash

# IMPLEMENTATIONS
serial_implementation() {
    echo "Compiling serial program..."
    if [ -n "$TESTING" ]; then
        g++ -g -Wall -DTESTING -o serial src/serial.cpp
    else
        g++ -g -Wall -o serial src/serial.cpp
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
    if [ -n "$TESTING" ]; then
        g++ -g -Wall -fopenmp -DTESTING -o omp src/omp.cpp
    else
        g++ -g -Wall -fopenmp -o omp src/omp.cpp
    fi
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

mpi_implementation() {
    echo "Compiling mpi program..."
    mpic++ -std=c++11 -o mpi src/mpi.cpp
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
    4)
        mpi_implementation
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

exit 0
