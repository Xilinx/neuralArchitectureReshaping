#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 SAFARI Research Group at ETH Zurich

import os
import sys 

sam_dir_path = sys.argv[1]
out_dir_path = sys.argv[2]


dir_list = os.listdir(sam_dir_path)

for dir in dir_list:
    test = dir.strip().split(".")
    os.system("samtools stats --threads 40 " + sam_dir_path + dir + " | grep ^SN | cut -f 2- > " + out_dir_path + test[0] + ".txt")
