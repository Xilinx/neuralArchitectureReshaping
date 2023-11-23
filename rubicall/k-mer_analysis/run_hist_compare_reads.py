import os
import sys 
from datetime import date
today = date.today()
import threading


if (HUMAN):
    reads=[
    '20201102_1746_6C_PAG07506_acad8a68'
    ]
    model=['causalcall','crf-fast','bonito','sacall','dorado','rubicall','crf-sup'] #fasta for human

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
    model=['causalcall','bonito','rubicall'] #fastq bacterial
    model=['crf-fast','crf-sup','dorado','sacall'] #fasta bacterial

BBMAP_PATH=""
READ_PATH=""
OUTPUT_PATH=""
HUMAN=True

def run_command(base_in_path,final_output_path,m,r,k):
    final_path=base_in_path +"/"+m+"/"+r
    command=BBMAP_PATH+"/khist.sh in=" +final_path+".fasta threads=256  k="+str(k)+" khist="+final_output_path+"/"+m+"/"+r+"_hist.txt peaks="+final_output_path+"/"+m+"/"+r+"_peak.txt"
    print(command)
    os.system(command)

def run_command_human(base_in_path,final_output_path,m,r,k):
    final_path=base_in_path +"/human/basecalled_reads_mix/"+m+"/"+r
    command=BBMAP_PATH+"/khist.sh in=" +final_path+".fasta threads=256 k="+str(k)+"   khist="+final_output_path+"/"+m+"/"+r+"_hist.txt peaks="+final_output_path+"/"+m+"/"+r+"_peak.txt"
    print(command)
    os.system(command)
   

k_value=[15]
for r in reads:    
    for m in model:
        for k in k_value:
            final_output_path=OUTPUT_PATH+"/k-mer_hist_compare_"+str(k)+"/basecalled_reads"
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
                run_command_human(READ_PATH,final_output_path,m,r,k)
            else:
                run_command(READ_PATH,final_output_path,m,r,k)
  
