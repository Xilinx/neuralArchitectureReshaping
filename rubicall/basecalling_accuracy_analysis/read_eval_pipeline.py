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

def read_results(output_path):    
        print("*****execute for = *****".format(output_path))
        command = "python3 basecalling_comparison/analysis_scripts/get_median_identity.py "+ output_path
        print("command:::", command)
        reads_median = (subprocess.getoutput(command))

        print([output_path,reads_median])

basecalled_reads_results=str(sys.argv[1]) #path to generated generated_basecalled_reads.tsv file
read_results(basecalled_reads_results)


