import os
import sys 
from datetime import date
import threading

BBMAP_PATH=""
ASSEMBLY_PATH=""
OUTPUT_PATH=""
HUMAN=True
k_value=[15]


if (HUMAN):
    reads=[
    '20201102_1746_6C_PAG07506_acad8a68'
    ]

else:
    reads=['acinetobacter',
    'haemophilus_haemolyticus_M1C132_1',
    'klebsiella_pneumoniae_NUH29',
    'klebsiella_pneumoniae_INF042',
    'klebsiella_pneumoniae_INF032',
    'stenotrophomonas_maltophilia_17_G_0092_Kos',
    'staphylococcus_aureus_CAS38_02_fast5s',
    'serratia_marcescens_17',
    'klebsiella_pneumoniae_KSB2_1B']

model=['bonito','causalcall','crf-fast','crf-sup','rubicall','sacall','dorado']

def run_command(final_output_path,m,r):
    final_path=ASSEMBLY_PATH +"/"+m+"/"+r+"/"+r+"_minimap2_contigs.fasta"
    command=BBMAP_PATH+"/kmercountexact.sh in=" +final_path+" out=stdout  fastadump=f k=15 | sort -k2,2rn  - > "+final_output_path+"/"+m+"/"+r+".out"
    print(command)
    os.system(command)

def run_command_human(final_output_path,m,r):
    final_path=ASSEMBLY_PATH +"/"+m+"/human/human_minimap2_contigs.fasta"
    command=BBMAP_PATH+"/kmercountexact.sh in=" +final_path+" out=stdout  fastadump=f k=15 | sort -k2,2rn  - > "+final_output_path+"/"+m+"/"+r+".out"
    print(command)
    os.system(command)


k=15
final_output_path=OUTPUT_PATH+"/k-mer_compare_"+str(k)+"/basecalled_assembly"
isExist = os.path.exists(final_output_path)
if not isExist:
                    # Create a new directory because it does not exist 
                    os.makedirs(final_output_path)
                    print("The new directory is created!")



for m in model:
    if not os.path.exists(final_output_path+"/"+m):
                    os.makedirs(final_output_path+"/"+m)
                    print("The new directory is created!")
    for r in reads:
        if(HUMAN):
                run_command_human(final_output_path,m,r,)
        else:
                run_command(final_output_path,m,r,)
