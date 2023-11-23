
import os
import sys 
from datetime import date
import threading


OUTPUT_PATH=""
HUMAN=True
ASSEMBLY_PATH=""
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


model=['crf-fast','bonito','sacall','dorado','rubicall','crf-sup','causalcall'] #fasta for human

def run_command(final_output_path,m,r,k):
    final_path=ASSEMBLY_PATH +"/"+m+"/"+r+"/"+r+"_minimap2_contigs.fasta"
    command="/mnt/batty/sgagandeep/bbmap/khist.sh in=" +final_path+" threads=256  k="+str(k)+" khist="+final_output_path+"/"+m+"/"+r+"_hist.txt peaks="+final_output_path+"/"+m+"/"+r+"_peak.txt"
    print(command)
    os.system(command)

def run_command_human(final_output_path,m,r,k):
    final_path=ASSEMBLY_PATH +"/"+m+"/human/human_minimap2_contigs.fasta"
    command="/mnt/batty/sgagandeep/bbmap/khist.sh in=" +final_path+" threads=256 k="+str(k)+"   khist="+final_output_path+"/"+m+"/"+r+"_hist.txt peaks="+final_output_path+"/"+m+"/"+r+"_peak.txt"
    print(command)
    os.system(command)
   

for k in k_value:
    for r in reads:    
        for m in model:
        
            final_output_path=OUTPUT_PATH+"/k-mer_hist_compare_"+str(k)+"/basecalled_assembly"
            isExist = os.path.exists(final_output_path)
            if not isExist:
                                # Create a new directory because it does not exist 
                                os.makedirs(final_output_path)
                                print("The new directory is created!")

            if not os.path.exists(final_output_path+"/"+m):
                        # Create a new directory because it does not exist 
                        os.makedirs(final_output_path+"/"+m)
                        print("The new directory is created!")

            if(HUMAN):
                run_command_human(final_output_path,m,r,k)
            else:
                run_command(final_output_path,m,r,k)
