import os
import sys 
from datetime import date
import subprocess
import threading
import csv


OUTPUT_PATH=""
HUMAN=True
READ_PATH=""

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
    command="seqkit stats "+READ_PATH+"/"+m+"/"+r+".fasta >"+final_output_path+".log"
    print(command)
    os.system(command)

def run_command_human(final_output_path,m,r):
    command="seqkit stats "+READ_PATH+"/"+m+"/"+r+".fasta  >"+final_output_path+".log"
    print(command)
    os.system(command)

def read_stats(output,name):  
     
        import re  
        print("*****execute for = *****".format(output))
        command = "cat "+output+".log |  tail -n 1 | awk  '{print $7}'"
        print("command:::", command)
        temp = (subprocess.getoutput(command))
        avg_len=(temp.replace(',', ''))
        avg_len = re.sub('"','',avg_len)

        output_file=name+".csv"
        if os.path.exists(output_file):
            print("files  exist")
            append_write = 'a' # append if already exists
            with open(output_file, append_write) as f:
                w = csv.writer(f)
                w.writerow([r,m,avg_len])

        
        else:
            append_write = 'w' # make a new file if not
            with open(output_file, append_write) as f:
                w = csv.writer(f)
                w.writerow(['Reads','Basecaller','avg_len'])
                w.writerow([r,m,avg_len])

        print([r,m,avg_len])


read_result="seqkit_stats"

for m in model:
    if not os.path.exists(output_path+"/"+m):
                    os.makedirs(output_path+"/"+m)
                    print("The new directory is created!")
    for r in reads:
        final_output_path=output_path +"/"+m+"/"+r
        x = threading.Thread(target=run_command_human,args=(final_output_path,m,r,))
        x.start()
        
# Read results
for r in reads:
    for m in model:
        final_output_path=output_path +"/"+m+"/"+r
        read_stats(final_output_path,read_result)
        
