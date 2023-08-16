# run_all.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

import os
import sys 
from datetime import date
today = date.today()

model_base_path="../models/"


modeltype_all=['bonito','rubicallmp','rubiconnoskipfp','crf-fast','crf-sup']
model_base_path_all=['bonito','rubicallmp','rubiconnoskipfp','crf-fast','crf-sup']


specie_dir_all=[
    "Acinetobacter_pittii_16-377-0801",
    "Haemophilus_haemolyticus_M1C132_1",
    "Klebsiella_pneumoniae_INF032",
    "Klebsiella_pneumoniae_INF042",
    "Klebsiella_pneumoniae_KSB2_1B",
    "Klebsiella_pneumoniae_NUH29",
    "Serratia_marcescens_17-147-1671",
    "Staphylococcus_aureus_CAS38_02",
    "Stenotrophomonas_maltophilia_17_G_0092_Kos"]


name_append = today.strftime("%m_%d_%Y")
for specie in specie_dir_all:
    output_path = "basecalled_read_"+str(name_append)
    isExist = os.path.exists(output_path)
    if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(output_path)
            print("The new directory is created!")
    for modeltype,final_path in zip(modeltype_all,model_base_path_all): 
        final_output_path = output_path+"/"+modeltype
        isExist = os.path.exists(final_output_path)
        if not isExist:
                # Create a new directory because it does not exist 
                os.makedirs(final_output_path)
                print("The new directory is created!")
        
        command="rubicon basecalling " + model_base_path+final_path +"  ../../data/organism/"+str(specie)+" --type "+str(modeltype)+ " > " + final_output_path+"/"+ specie + ".fasta"
        print(command)
        os.system(command)
        final_output_path=""
