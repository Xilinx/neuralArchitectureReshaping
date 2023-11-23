#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

#!/bin/bash

BASE_DIR=$1

mkdir -p ${BASE_DIR}/dnadiff
for i in `echo ${BASE_DIR}/*.fasta`; do 
    fname=`basename $i | sed s/.fasta/_dnadiff/g`; 
    
    # Align using nucmer with multi-threading (e.g., 32 threads)
    nucmer --maxmatch -l 100 -c 500 -t 32 -p ${BASE_DIR}/dnadiff/$fname ../ref_paternal.fa $i
    
    # Run dnadiff on the resulting delta file
    dnadiff -d ${BASE_DIR}/dnadiff/$fname.delta -p ${BASE_DIR}/dnadiff/$fname > ${BASE_DIR}/dnadiff/$fname.stdout 2> ${BASE_DIR}/dnadiff/$fname.stderr;
done

echo 'Dnadiff results:';
for i in `echo ${BASE_DIR}/dnadiff/*.report`; do 
    echo $i; awk '{if(NR == 24){print "Avg. Identity: "$3}else if(NR == 12){print "Genome Fraction: "$2}}' $i;
done
