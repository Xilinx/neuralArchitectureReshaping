#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

#!/bin/bash

THREAD=32
SLURM_OPTIONS="--mem=200000"

#We use SLURM to run our commands. To run them without SLURM, you can run the 'command' in --wrap="command" in bash. To use them with SLURM, please configure SLURM_OPTIONS and THREAD above based on your system configuration.
sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../dorado/human/slurm_dnadiff.out -e ../dorado/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../dorado/human/"

sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../bonito/human/slurm_dnadiff.out -e ../bonito/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../bonito/human/"

sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../causalcall/human/slurm_dnadiff.out -e ../causalcall/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../causalcall/human/"

sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../rubicallfp/human/slurm_dnadiff.out -e ../rubicallfp/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../rubicallfp/human/"

sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../crf-fast/human/slurm_dnadiff.out -e ../crf-fast/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../crf-fast/human/"

sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../crf-sup/human/slurm_dnadiff.out -e ../crf-sup/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../crf-sup/human/"

sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../rubicallmp/human/slurm_dnadiff.out -e ../rubicallmp/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../rubicallmp/human/"

sbatch -c ${THREAD} "${SLURM_OPTIONS}" -o ../sacall/human/slurm_dnadiff.out -e ../sacall/human/slurm_dnadiff.err --wrap="./analyze_assembly.sh ../sacall/human/"
