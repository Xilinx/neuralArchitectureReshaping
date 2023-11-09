# run_basecalling.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

import os
import sys 
from datetime import date
today = date.today()

specie_dir=str(sys.argv[1])
model_dir=str(sys.argv[2])
modeltype=str(sys.argv[3])
output_path=str(os.getcwd())
              
command="rubicon basecalling " + model_dir +"  "+str(specie_dir)+" --type "+str(modeltype)+ " > " + output_path+"/"+ modeltype + ".fasta"
print(command)
os.system(command)
