# __init__.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from rubicon.basemodule import download,train,  qabas , prune, skipclip , basecalling

modules = [
    'download','train', 'qabas' ,'prune' , 'skipclip','basecalling'
]

__version__ = '0.1'


def main():
    parser = ArgumentParser(
        'rubicon',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s {}'.format(__version__)
    )

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)
