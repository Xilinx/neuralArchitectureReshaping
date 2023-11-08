#!/bin/bash
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

THREAD=32

#haemophilus_haemolyticus_M1C132_1
OUTDIR="./haemophilus_haemolyticus_M1C132_1/"
READS="../../basecalled_reads/crf-fast/haemophilus_haemolyticus_M1C132_1.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="haemophilus_haemolyticus_M1C132_1"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"


#acinetobacter
OUTDIR="./acinetobacter/"
READS="../../basecalled_reads/crf-fast/acinetobacter.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="acinetobacter"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"


#klebsiella_pneumoniae_INF032
OUTDIR="./klebsiella_pneumoniae_INF032/"
READS="../../basecalled_reads/crf-fast/klebsiella_pneumoniae_INF032.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="klebsiella_pneumoniae_INF032"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"

#klebsiella_pneumoniae_INF042
OUTDIR="./klebsiella_pneumoniae_INF042/"
READS="../../basecalled_reads/crf-fast/klebsiella_pneumoniae_INF042.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="klebsiella_pneumoniae_INF042"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"

#klebsiella_pneumoniae_KSB2_1B
OUTDIR="./klebsiella_pneumoniae_KSB2_1B/"
READS="../../basecalled_reads/crf-fast/klebsiella_pneumoniae_KSB2_1B.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="klebsiella_pneumoniae_KSB2_1B"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"

#klebsiella_pneumoniae_NUH29
OUTDIR="./klebsiella_pneumoniae_NUH29/"
READS="../../basecalled_reads/crf-fast/klebsiella_pneumoniae_NUH29.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="klebsiella_pneumoniae_NUH29"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"

#serratia_marcescens_17
OUTDIR="./serratia_marcescens_17/"
READS="../../basecalled_reads/crf-fast/serratia_marcescens_17.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="serratia_marcescens_17"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"

#staphylococcus_aureus_CAS38_02_fast5s
OUTDIR="./staphylococcus_aureus_CAS38_02_fast5s/"
READS="../../basecalled_reads/crf-fast/staphylococcus_aureus_CAS38_02_fast5s.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="staphylococcus_aureus_CAS38_02_fast5s"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"

#stenotrophomonas_maltophilia_17_G_0092_Kos
OUTDIR="./stenotrophomonas_maltophilia_17_G_0092_Kos/"
READS="../../basecalled_reads/crf-fast/stenotrophomonas_maltophilia_17_G_0092_Kos.fasta"
PRESET="ava-ont"

mkdir -p ${OUTDIR}

PREFIX="stenotrophomonas_maltophilia_17_G_0092_Kos"
sbatch -p bio -c 32 -o ${OUTDIR}/${PREFIX}.out -e ${OUTDIR}/${PREFIX}.err --wrap="../minimap2-overlap.sh ${OUTDIR} ${PREFIX} ${READS} ${PRESET} ${THREAD}"

