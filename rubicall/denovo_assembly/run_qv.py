import os
import sys 
from datetime import date
import subprocess
import threading
import csv

OUTPUT_PATH=""
HUMAN=True
READ_PATH=""
ASSEMBLY_PATH=""
REFERENCE_PATH=""
INSPECTOR_PATH=""

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

    model=['crf-fast','crf-sup','dorado','sacall'] #fasta bacterial
    model=['causalcall','bonito','rubicall'] #fastq bacterial


def run_command(final_output_path,m,r):
    command=INSPECTOR_PATH+"/inspector.py -c \
    "+ASSEMBLY_PATH+"/"+m+"/"+r+"/"+r+"_minimap2_contigs.fasta \
    -r "+READ_PATH+"/"+m+"/"+r+".fastq \
    --ref "+REFERENCE_PATH+"/reference_"+r+".fasta -t 16 --datatype nanopore -o "+final_output_path
    # command="/mnt/batty/sgagandeep/bbmap/kmercountexact.sh in=" +final_path+" out=stdout  fastadump=f | sort -k2,2rn  - > "+final_output_path+"/"+m+"/"+r+"_fasta.out"
    print(command)
    os.system(command)

def run_command_human(final_output_path,m,r):
    command=INSPECTOR_PATH+"/inspector.py -c \
   "+ASSEMBLY_PATH+"/"+m+"/human/human_minimap2_contigs.fasta \
    -r "+READ_PATH+"/human/basecalled_reads_mix/"+m+"/"+r+".fasta \
    --ref "+REFERENCE_PATH+"/reference_"+r+".fasta -t 128 --datatype nanopore  -o "+final_output_path
    # command="/mnt/batty/sgagandeep/bbmap/kmercountexact.sh in=" +final_path+" out=stdout  fastadump=f | sort -k2,2rn  - > "+final_output_path+"/"+m+"/"+r+"_fasta.out"
    print(command)
    os.system(command)

def read_qv(output,name):    
        print("*****execute for = *****".format(output))
        command = "cat "+output+"/summary_statistics | grep QV | awk  '{print $2}'"
        print("command:::", command)
        qv_value = (subprocess.getoutput(command))


        output_file=""+name+".csv"
        if os.path.exists(output_file):
            print("files  exist")
            append_write = 'a' # append if already exists
            with open(output_file, append_write) as f:
                w = csv.writer(f)
                w.writerow([r,m,qv_value])

        
        else:
            append_write = 'w' # make a new file if not
            with open(output_file, append_write) as f:
                w = csv.writer(f)
                w.writerow(['Reads','Basecaller','QV'])
                w.writerow([r,m,qv_value])

        print([r,m,qv_value])


read_result="qv"

for m in model:
    if not os.path.exists(OUTPUT_PATH+"/"+m):
                    os.makedirs(OUTPUT_PATH+"/"+m)
                    print("The new directory is created!")
    for r in reads:
        final_output_path=OUTPUT_PATH +"/"+m+"/"+r
        if(HUMAN):
            x = threading.Thread(target=run_command_human,args=(final_output_path,m,r,))
        else:
            x = threading.Thread(target=run_command,args=(final_output_path,m,r,))
        x.start()
        
 # Read results       
for r in reads:
    for m in model:
        final_output_path=OUTPUT_PATH +"/"+m+"/"+r
        read_qv(final_output_path,read_result)
        
