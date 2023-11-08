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


plt.style.use("grayscale")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['axes.labelsize'] = 6.5
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 5
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


with open('read_identity.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter='\t')
        next(plots)
        for row in plots:
            print(row)
            species.append(str(row[0]))
            Causallcall.append(float(row[1]))
            Guppy.append(float(row[2]))
            Guppy_fast.append(float(row[3]))
            Bonito.append(float(row[4]))
            RUBICON_AIE.append(float(row[5]))


apps_number=10

conf = configs
temp = configs
configs =[]
fig=plt.figure(figsize=(4.3,1))
ax = plt.subplot(111)
plt.ylabel('Basecalling\nAccuracy (%)', labelpad=0.1)

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
plt.legend(bbox_to_anchor=(0.85, 1.2),loc='upper right', ncol=7 , \
frameon=True,markerscale=1,prop=legend_properties, framealpha=0.4, borderaxespad=0,handletextpad=0.2, handlelength=0.8, columnspacing=1)

plt.axvspan(18.4,21,color='#999999', alpha=0.5)
ax.set_xlim([-0.5, 20.5])
ax.set_ylim([70,100])
# plt.yticks([1, 2, 3], ['10', '100', '1000'])
ax.tick_params(axis='y', which='major', pad=0.6)
ax.tick_params(axis='x', which='major', pad=0.6)

plt.savefig("read_identity_aie_CRF_2"+".pdf", dpi=600, format='pdf', transparent=False, bbox_inches='tight')
