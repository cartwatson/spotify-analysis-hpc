#!/bin/bash

# Function to find Python command
find_python_command() {
    if command -v python3 &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        echo "Python not found, please install Python."
        exit 1
    fi
}

# Define the URL
URL="http://localhost:8000/src/python/plotly_visualization.html"

# Change directory to the root of your project
cd ../

# Find Python command
PYTHON_CMD=$(find_python_command)

# Start Python server in the background
$PYTHON_CMD -m http.server &
SERVER_PID=$!

# Function to kill the server
kill_server() {
    echo "Stopping server..."
    kill $SERVER_PID
}

# Trap to execute the kill_server function when the script exits
trap kill_server EXIT

# Wait for a moment to ensure the server starts
sleep 2

# Detect the operating system and open the browser accordingly
case "$(uname)" in
    "Linux") xdg-open $URL;;
    "Darwin") open $URL;; # For MacOS
    "CYGWIN"*|"MINGW"*|"MSYS"*) start $URL;; # For Windows environments in Cygwin, MinGW, or MSYS
    *)
      echo "Unsupported OS: $(uname)"
      exit 1
      ;;
esac

# Keep the script running until manually terminated
wait $SERVER_PID
