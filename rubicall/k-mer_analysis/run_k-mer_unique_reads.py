import os
import sys 
from datetime import date
import threading

BBMAP_PATH=""
READ_PATH=""
OUTPUT_PATH=""
HUMAN=True
k_value=[15]


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





def run_command(base_in_path,final_output_path,m,r,k):
    final_path=base_in_path +"/"+m+"/"+r
    command=BBMAP_PATH+"/kmercountexact.sh in=" +final_path+".fastq threads=256 k="+str(k)+"   out=stdout  fastadump=f  | sort -k2,2rn  - > "+final_output_path+"/"+m+"/"+r+".out"
    
    print(command)
    os.system(command)

def run_command_human(base_in_path,final_output_path,m,r,k):
    final_path=base_in_path +"/human/basecalled_reads_mix/"+m+"/"+r
    command=BBMAP_PATH+"/kmercountexact.sh in=" +final_path+".fasta threads=256 k="+str(k)+"    out=stdout  fastadump=f  | sort -k2,2rn  - > "+final_output_path+"/"+m+"/"+r+".out"
    print(command)
    os.system(command)
   



for r in reads:    
    for m in model:
        for k in k_value:
            final_output_path=OUTPUT_PATH+"/k-mer_compare_"+str(k)+"/basecalled_reads"
            # hist_final_output_path=output_path+"/k-mer_hist_compare_"+str(k)+"/basecalled_reads"
            isExist = os.path.exists(final_output_path)
            if not isExist:
                                # Create a new directory because it does not exist 
                                os.makedirs(final_output_path)
                                print("The new directory is created!")
    
            if not os.path.exists(final_output_path+"/"+m):
                        # Create a new directory because it does not exist 
                        os.makedirs(final_output_path+"/"+m)
                        print("The new directory is created!")
            run_command_human(READ_PATH,final_output_path,m,r,k)
            # run_command(base_in_path,final_output_path,m,r,k)
        # run_command(base_in_path,final_output_path,m,r,)
        # x = threading.Thread(target=run_command,args=(base_in_path,final_output_path,m,r,))
        # x.start()
        # os.system(command)
# final_output_path=output_path+"/k-mer_compare_15/basecalled_reads"
# isExist = os.path.exists(final_output_path)
# if not isExist:
#                     # Create a new directory because it does not exist 
#                     os.makedirs(final_output_path)
#                     print("The new directory is created!")



# for m in model:
#     if not os.path.exists(final_output_path+"/"+m):
#                     # Create a new directory because it does not exist 
#                     os.makedirs(final_output_path+"/"+m)
#                     print("The new directory is created!")
#     for r in reads:
#         # run_command_human(base_in_path,final_output_path,m,r)
#         x = threading.Thread(target=run_command,args=(base_in_path,final_output_path,m,r,))
#         x.start()
#         # os.system(command)

