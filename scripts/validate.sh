#!/bin/bash

# run serial implementation to compare other implementations against
echo "Compiling serial program..."
g++ -g -Wall -o serial src/serial.cpp
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi
echo "Running serial program..."
./serial
mv src/data/output.csv src/data/serial_output.csv # make sure serial output is not overwriten

# TODO: run other implementations and compare output to serial output

echo -e "\n--------------------comparison results--------------------"
# output results of comparisons
if cmp -s src/data/serial_output.csv src/data/cpu-shared-output.csv; then
    echo "CPU Shared Memory implementation is correct"
else
    echo "Error in CPU Shared Memory implementation"
fi
echo -e "----------------------------------------------------------\n"

echo "Cleaning up..."
rm serial
rm src/data/serial_output.csv
rm src/data/cpu-shared-output.csv
