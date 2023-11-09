#!/bin/bash

echo "Compiling serial program..."
g++ -g -Wall -o serial serial.cpp
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

echo "Running serial program..."
if [ $1 ]; then
    ./serial $1
else
    ./serial
fi
