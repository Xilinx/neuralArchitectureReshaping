#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

#!/bin/bash

BASE_DIR=$1

mkdir -p ${BASE_DIR}/quast
quast --large -t 32 -m 0 -o ${BASE_DIR}/quast -r ../ref_paternal.fa -k ${BASE_DIR}/*.fasta > ${BASE_DIR}/quast/quast.stdout 2> ${BASE_DIR}/quast/quast.stderr

echo;
echo ${BASE_DIR}
echo 'QUAST results:';
awk -F '\t' '{
    if(NR == 1){
        for(i = 2; i <= NF; i++){
            tool[i-2] = $i
        }
    }
    if($1 == "Average GC (%)"){
        print ""
        print "Average GC (%):"
        for(i = 2; i <= NF; i++){
            print tool[i-2]": "$i
        }
    } else if($1 == "Total length"){
        print ""
        print "Assembly Length:"
        for(i = 2; i <= NF; i++){
            print tool[i-2]": "$i
        }
    }else if($1 == "NG50"){
        print ""
        print "NG50:"
        for(i = 2; i <= NF; i++){
            print tool[i-2]": "$i
        }
    }
}' ./${BASE_DIR}/quast/report.tsv

