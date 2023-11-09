#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

#!/bin/bash

THREAD=1

SLURM_OPTIONS="-p bio"

#We use SLURM to run our commands. To run them without SLURM, you can run the 'command' in --wrap="command" in bash. To use them with SLURM, please configure SLURM_OPTIONS and THREAD above based on your system configuration.
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./acinetobacter/slurm_dnadiff.out -e ./acinetobacter/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./acinetobacter/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./haemophilus_haemolyticus_M1C132_1/slurm_dnadiff.out -e ./haemophilus_haemolyticus_M1C132_1/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./haemophilus_haemolyticus_M1C132_1/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./klebsiella_pneumoniae_INF032/slurm_dnadiff.out -e ./klebsiella_pneumoniae_INF032/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./klebsiella_pneumoniae_INF032/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./klebsiella_pneumoniae_INF042/slurm_dnadiff.out -e ./klebsiella_pneumoniae_INF042/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./klebsiella_pneumoniae_INF042/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./klebsiella_pneumoniae_KSB2_1B/slurm_dnadiff.out -e ./klebsiella_pneumoniae_KSB2_1B/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./klebsiella_pneumoniae_KSB2_1B/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./klebsiella_pneumoniae_NUH29/slurm_dnadiff.out -e ./klebsiella_pneumoniae_NUH29/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./klebsiella_pneumoniae_NUH29/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./serratia_marcescens_17/slurm_dnadiff.out -e ./serratia_marcescens_17/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./serratia_marcescens_17/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./staphylococcus_aureus_CAS38_02_fast5s/slurm_dnadiff.out -e ./staphylococcus_aureus_CAS38_02_fast5s/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./staphylococcus_aureus_CAS38_02_fast5s/"
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ./stenotrophomonas_maltophilia_17_G_0092_Kos/slurm_dnadiff.out -e ./stenotrophomonas_maltophilia_17_G_0092_Kos/slurm_dnadiff.err --wrap="./analyze_assembly.sh ./stenotrophomonas_maltophilia_17_G_0092_Kos/"
