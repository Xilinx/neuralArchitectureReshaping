# download.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

"""
Download
"""
import sys
import os
import re
from shutil import rmtree
from zipfile import ZipFile
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tarfile
from rubicon.util import __data__, __models__,__dataorg__
from bonito.cli.convert import main as convert
from bonito.cli.convert import argparser as cargparser

import requests
from tqdm import tqdm



class File:
    """
    Small class for downloading models and training assets.
    """
    __url__ = "https://nanoporetech.box.com/shared/static/"
    __org_url__ = "https://bridges.monash.edu/ndownloader/files/"
    def __init__(self, path, url_frag, force=False, org_url=False):
        self.path = path
        self.force = force
        if org_url:
            self.url = os.path.join(self.__org_url__, url_frag)
        else:
            self.url = os.path.join(self.__url__, url_frag)

    def location(self, filename):
        return os.path.join(self.path, filename)

    def exists(self, filename):
        return os.path.exists(self.location(filename))

    def download(self):
        """
        Download the remote file
        """
        # create the requests for the file
        req = requests.get(self.url, stream=True)
        total = int(req.headers.get('content-length', 0))
        fname = re.findall('filename="([^"]+)', req.headers['content-disposition'])[0]

        # skip download if local file is found
        if self.exists(fname.strip('.zip')) and not self.force:
            print("[skipping %s]" % fname)
            return
      
        if self.exists(fname.strip('.zip')) and self.force:
            rmtree(self.location(fname.strip('.zip')))

        # download the file
        with tqdm(total=total, unit='iB', ascii=True, ncols=100, unit_scale=True, leave=False) as t:
            with open(self.location(fname), 'wb') as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        print("[downloaded %s]" % fname)

        # unzip .zip files
        if fname.endswith('.zip'):
            with ZipFile(self.location(fname), 'r') as zfile:
                zfile.extractall(self.path)
            os.remove(self.location(fname))

        # unzip .gz files
        if fname.endswith('.gz'):
            file=tarfile.open(self.location(fname))
            file.extractall(self.path+"/"+fname.strip('_fast5s.tar.gz'))
            file.close()
        # convert chunkify training files
        if fname.endswith('.hdf5'):
            print("[converting %s]" % fname)
            args = cargparser().parse_args([
                self.location(fname),
                self.location(fname).strip('.hdf5')
            ])
            convert(args)

organism = {
    # Read from: https://bridges.monash.edu/articles/dataset/Raw_fast5s/7676174
    # https://doi.org/10.26180/5c5a5fa08bbee
    
    "Acinetobacter_pittii_16-377-0801": "14260511",
    "Haemophilus_haemolyticus_M1C132_1": "14260514",
    "Klebsiella_pneumoniae_INF032": "15188573",
    "Klebsiella_pneumoniae_INF042": "14260517",
    "Klebsiella_pneumoniae_NUH29": "14260550",
    "Klebsiella_pneumoniae_KSB2_1B": "14260541",
    "Staphylococcus_aureus_CAS38_02": "14260568",
    "Shigella_sonnei_2012-02037": "14260562",
    "Serratia_marcescens_17-147-1671": "14260556",
    "Stenotrophomonas_maltophilia_17_G_0092_Kos": "14260574",
}

training = [
    "cmh91cxupa0are1kc3z9aok425m75vrb.hdf5",
]


def main(args):
    """
    Download training sets
    """
    if args.training or args.all:
        print("[downloading ONT data]")
        for train in training:
            File(__data__, train, args.force).download()

    if args.organism or args.all:
        
        if args.show:
            print("[available organism]", file=sys.stderr)
            for org in organism:
                print(f" - {org}", file=sys.stderr)
        else:
            print("[downloading organism]", file=sys.stderr)
            for org in organism.values():
                File(__dataorg__, org,args.force,org_url=True).download()


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true')
    group.add_argument('--training', action='store_true')
    group.add_argument('--organism', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('--list', '--show', dest='show', action='store_true')
    return parser