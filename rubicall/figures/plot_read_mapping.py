#!/bin/bash
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.ticker as ticker
import mpl_toolkits.axisartist as AA
import matplotlib.font_manager
import pandas as pd

BASES_MAPPED=True
MISMATCH=False
READ_MAP=False
READ_UNMAP=False

plt.style.use("grayscale")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelsize'] = 6.5
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 4
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
bar_width=0.35
patterns = [ "|" , "\\" , "/" , "+" , "-", ".", "*","x", "o", "O" ]

colors=["#F5D491","#94A578","#D1F0B1","#5293A3","#FF8F85","#9B382E"]

df = pd.read_csv("basecalling_speed.csv", delimiter='\t')

configs =list(df.columns) 
del configs[0]

config_number = 4

species=[]
Causallcall = []
Guppy = []
Guppy_fast=[]
Bonito=[]
RUBICON=[]
RUBICON_AIE=[]

if(BASES_MAPPED):
    resFile='mapped_bases.csv'

if(MISMATCH):
    resFile='mismatch.csv'

if(READ_MAP):
    resFile='read_mapped.csv'

if(READ_UNMAP):
    resFile='read_unmapped.csv'

with open(resFile,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        next(plots)
        for row in plots:
            print(row)
            species.append(str(row[0]))
            Causallcall.append(float(row[1]))
            Guppy_fast.append(float(row[2]))
            Bonito.append(float(row[3]))
            RUBICON_AIE.append(float(row[4]))
# if(READ_UNMAP):
#     Causallcall = np.log10(np.array(Causallcall))
#     Guppy_fast = np.log10(np.array(Guppy_fast))
#     Bonito = np.log10(np.array(Bonito))
#     RUBICON_AIE = np.log10(np.array(RUBICON_AIE))
#     apps_number=10

conf = configs
temp = configs
configs =[]
fig=plt.figure(figsize=(4.3,1))
ax = plt.subplot(111)
if(BASES_MAPPED):
    label='Bases Mapped'
if(MISMATCH):
    label='Mismatches'
   
if(READ_MAP):
    label='Reads Mapped'

if(READ_UNMAP):
    label='Reads Unmapped'
plt.ylabel(label, labelpad=0.1)

ind = np.arange(start = 0, stop = 21, step = 2.1)

ax.yaxis.grid(color = "grey",linewidth=0.5,zorder=1)
ypos = -1.8

plt.xticks(ind+config_number/2*bar_width,species,fontsize=6,rotation=35,ha='right')

print("first",ind)
# ax.set_xticks(ind+8*bar_width)
ax.bar(ind,
                    # using pre_rel data
                    Causallcall,
                    # labeled
                    # with alpha
                    alpha=0.9,
                    # with color
                    label="Causalcall",
                    color=colors[0],
                    # with bar width
                    width=bar_width,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=2
                    # ,
                    #  hatch=patterns[4]
                    # with border color
                    )
# ax.bar(ind+bar_width,
#                     # using pre_rel data
#                     Guppy,
#                     # labeled
#                     # with alpha
#                     alpha=0.9,
#                     # with color
#                     label="Guppy",
#                     color=colors[1],
#                     # with bar width
#                     width=bar_width,
#                     edgecolor='black',
#                    linewidth=0.4,
#                     zorder=2
#                     # with border color
#                     )

print("second",ind)
ax.bar(ind+1*bar_width,
                    # using pre_rel data
                    Guppy_fast,
                    # labeled
                    # with alpha
                    alpha=0.9,
                    # with color
                    label="Bonito_CRF-fast",
                    color=colors[2],
                    # with bar width
                    width=bar_width,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=2
                    # with border color
                    )

print("third",ind)
ax.bar(ind + 2*bar_width,
                    # using pre_rel data
                    Bonito,
                    # labeled
                    # with alpha
                    alpha=0.9,
                    # with color
                    label="Bonito_CTC",
                    color="#b299c7",
                    # color="#b4dcf0",
                    # with bar width
                    width=bar_width,
                    edgecolor='black',
                   linewidth=0.5,
                    zorder=2
                    # with border color
                    )
ax.bar(ind + 3*bar_width,
                    # using pre_rel data
                    RUBICON_AIE,
                    # labeled
                    # with alpha
                    alpha=0.9,
                    # with color
                    label="RUBICALL",
                    color=colors[3],
                    # with bar width
                    width=bar_width,
                    edgecolor='black',
                   linewidth=0.5,
                    zorder=2
                    # with border color
                    )
# ax.bar(ind + 5*bar_width,
#                     # using pre_rel data
#                     RUBICON_AIE,
#                     # labeled
#                     # with alpha
#                     alpha=0.9,
#                     # with color
#                     label="RUBICON_AIE_MP",
#                     color=colors[4],
#                     # with bar width
#                     width=bar_width,
#                     edgecolor='black',
#                    linewidth=0.5,
#                     zorder=2
#                     # with border color
#                     )

ax.set_xticklabels(["Acinetobacter pittii\n16-377-0801",
"Haemophilus haemolyticus\nM1C132_1",
"Klebsiella pneumoniae\nINF032",
"Klebsiella pneumoniae\nINF042",
"Klebsiella pneumoniae\nKSB2_1B",
"Klebsiella pneumoniae\nNUH29",
"Serratia marcescens\n17-147-1671",
"Staphylococcus aureus\nCAS38_02",
"Stenotrophomonas maltophilia\n17_G_0092_Kos",
"Average"], rotation=45) 

import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
legend_properties = {'weight':'bold',"size":5}
handles, labels = ax.get_legend_handles_labels()

if (READ_UNMAP):
    plt.legend(bbox_to_anchor=(0.90, 1.2),loc='upper right', ncol=7 , \
frameon=True,markerscale=1,prop=legend_properties, framealpha=0.4, borderaxespad=0,handletextpad=0.2, handlelength=0.8, columnspacing=1)
else:
    plt.legend(bbox_to_anchor=(0.90, 1.2),loc='upper right', ncol=7 , \
    frameon=True,markerscale=1,prop=legend_properties, framealpha=0.4, borderaxespad=0,handletextpad=0.2, handlelength=0.8, columnspacing=1)

plt.axvspan(18.4,21,color='#999999', alpha=0.5)
ax.set_xlim([-0.5, 20.5])
if(BASES_MAPPED):
    ax.set_ylim([0,600000000])
    plt.text(0.2, 500000000, '(b)',fontsize=8,weight='bold')
if(MISMATCH):
    ax.set_ylim([0,100000000])
    plt.text(0.2, 82500000, '(a)',fontsize=8,weight='bold')
if(READ_MAP):
    ax.set_ylim([0,20000])
    plt.text(0.2, 16500, '(c)',fontsize=8,weight='bold')

if(READ_UNMAP):
    ax.text(0.2,22,(round(Guppy_fast[0],2)),fontsize=4.5,rotation=60)

    ax.text(1.75,310,(round(Causallcall[1],1)),fontsize=4.5,rotation=90)
    ax.text(2.1,310,(round(Guppy_fast[1],1)),fontsize=4.5,rotation=90)
    ax.text(2.5,310,(round(Bonito[1],1)),fontsize=4.5,rotation=90)
    ax.text(2.9,310,(round(RUBICON_AIE[1],1)),fontsize=4.5,rotation=90)
    
    ax.text(3.9,22,(round(Causallcall[2],2)),fontsize=4.5,rotation=60)
    ax.text(4.3,22,(round(Guppy_fast[2],2)),fontsize=4.5,rotation=60)
    ax.text(4.7,22,(round(Bonito[2],2)),fontsize=4.5,rotation=60)
    ax.text(5.1,22,(round(RUBICON_AIE[2],2)),fontsize=4.5,rotation=60)
    
    ax.text(6.5,30,(round(Guppy_fast[3],2)),fontsize=4.5,rotation=60)


    ax.text(14.5,22,round(Causallcall[7],2),fontsize=4.5,rotation=60)
    ax.text(14.9,22,round(Guppy_fast[7],2),fontsize=4.5,rotation=60)
    ax.text(15.3,22,round(Bonito[7],2),fontsize=4.5,rotation=60)
    ax.text(15.7,22,round(RUBICON_AIE[7],2),fontsize=4.5,rotation=60)
    plt.text(0.2, 250, '(d)',fontsize=8,weight='bold')
    ax.text(18.6,310,round(Causallcall[9],1),fontsize=4.5,rotation=90)
    ax.set_ylim([0,300])
    
    
# plt.yticks([1, 2, 3], ['10', '100', '1000'])
ax.tick_params(axis='y', which='major', pad=0.6)
ax.tick_params(axis='x', which='major', pad=0.6)
if(BASES_MAPPED):
    plt.savefig("samtools_mapped_bases_CRF_1"+".pdf", dpi=800, format='pdf', transparent=False, bbox_inches='tight')

if(MISMATCH):
    plt.savefig("samtools_mismatch_CRF_1"+".pdf", dpi=800, format='pdf', transparent=False, bbox_inches='tight')

if(READ_MAP):
    plt.savefig("samtools_read_map_CRF_1"+".pdf", dpi=800, format='pdf', transparent=False, bbox_inches='tight')


if(READ_UNMAP):
    plt.savefig("samtools_read_unmap_CRF_1"+".pdf", dpi=800, format='pdf', transparent=False, bbox_inches='tight')
