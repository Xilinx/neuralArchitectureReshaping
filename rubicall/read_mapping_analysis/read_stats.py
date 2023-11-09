#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

import os
import sys 
import csv

samtools_out_dir_path = "samtools_out_new/"

models = ["causalcall", "crf-fast", "bonito", "rubicon"]
organisms = ["acinetobacter", "haemophilus_haemolyticus_M1C132_1", "klebsiella_pneumoniae_INF032", "klebsiella_pneumoniae_INF042", "klebsiella_pneumoniae_KSB2_1B", "klebsiella_pneumoniae_NUH29", "serratia_marcescens_17", "staphylococcus_aureus_CAS38_02_fast5s", "stenotrophomonas_maltophilia_17_G_0092_Kos"]

file_reads_mapped = []
file_reads_unmapped = []
file_non_primary_alignments = []
file_bases_mapped = []
file_bases_mapped_cigar = []
file_mismatches = []
file_average_quality = []


for organism in organisms:
    file_reads_mapped.append([])
    file_reads_unmapped.append([])
    file_non_primary_alignments.append([])
    file_bases_mapped.append([])
    file_bases_mapped_cigar.append([])
    file_mismatches.append([])
    file_average_quality.append([])

    file_reads_mapped[-1].append(organism)
    file_reads_unmapped[-1].append(organism)
    file_non_primary_alignments[-1].append(organism)
    file_bases_mapped[-1].append(organism)
    file_bases_mapped_cigar[-1].append(organism)
    file_mismatches[-1].append(organism)
    file_average_quality[-1].append(organism)

    for model in models:
        file = open(samtools_out_dir_path + model + "/" + organism + ".txt")
        for line in file:
            if "reads mapped:" in line:
                word = int(line.strip().split(":")[1].strip().split("#")[0].strip())
                file_reads_mapped[-1].append(word)
            if "reads unmapped:" in line:
                word = int(line.strip().split(":")[1].strip().split("#")[0].strip())
                file_reads_unmapped[-1].append(word)
            if "non-primary alignments:" in line:
                word = int(line.strip().split(":")[1].strip().split("#")[0].strip())
                file_non_primary_alignments[-1].append(word)
            if "bases mapped:" in line:
                word = int(line.strip().split(":")[1].strip().split("#")[0].strip())
                file_bases_mapped[-1].append(word)
            if "bases mapped (cigar):" in line:
                word = int(line.strip().split(":")[1].strip().split("#")[0].strip())
                file_bases_mapped_cigar[-1].append(word)
            if "mismatches:" in line:
                word = int(line.strip().split(":")[1].strip().split("#")[0].strip())
                file_mismatches[-1].append(word)
            if "average quality:" in line:
                word = line.strip().split(":")[1].strip().split("#")[0].strip()
                file_average_quality[-1].append(word)


header = ["reads_mapped"]
for m in models:
    header.append(m)

with open("reads_mapped.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(file_reads_mapped)


header = ["reads_unmapped"]
for m in models:
    header.append(m)

with open("reads_unmapped.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(file_reads_unmapped)

header = ["non primary alignments"]
for m in models:
    header.append(m)

with open("non_primary_alignments.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(file_non_primary_alignments)


header = ["bases mapped"]
for m in models:
    header.append(m)

with open("bases_mapped.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(file_bases_mapped)

header = ["bases mapped (cigar)"]
for m in models:
    header.append(m)

with open("bases_mapped_cigar.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(file_bases_mapped_cigar)

header = ["mismatches"]
for m in models:
    header.append(m)

with open("mismatches.csv", 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(file_mismatches)

