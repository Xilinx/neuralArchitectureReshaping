#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 file1 file2 output_file"
    exit 1
fi

# Assign input arguments to variables
file1=$1
file2=$2
output_file=$3

# Sort the files and find common lines
comm -12 <(sort "$file1") <(sort "$file2") > "$output_file"

