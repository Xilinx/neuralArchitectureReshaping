#!/bin/bash

# Assigning the first input argument to 'thr' and the second to 'filename'
thr=$1
filename=$2

# Using awk to process the file and output to a new file
echo $filename
awk -v thr="$thr" '{if ($2 < thr) print $1}' "$filename" > $filename.under

