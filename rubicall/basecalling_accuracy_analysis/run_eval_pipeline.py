#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

import os
import subprocess
# import numpy as np
import os.path
import fileinput
from shutil import copyfile
import csv
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
from threading import Thread
from os import walk
import sys

def run_evaluation_pipeline(basecalled_reads):    
    print("*****execute for = *****{}".format(os.getcwd()))
    # Check whether the specified path exists or not
    log_path=os.getcwd()
    print("log_path:::",log_path)
    isExist = os.path.exists(log_path)

    command="./basecalling_comparison/analysis_scripts/analysis.sh "+basecalled_reads+" \
    2>&1 | tee "+log_path+"/"+basecalled_reads+".log"
    print(command)
    os.system(command)




basecalled_reads=str(sys.argv[1]) #path to basecalled reads
run_evaluation_pipeline(basecalled_reads)



