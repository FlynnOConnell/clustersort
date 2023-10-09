# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations

import argparse

import numpy as np

from clustersort.main import run
from clustersort.spk_config import SortConfig


def add_args(parser: argparse.ArgumentParser):
    """
    Adds suite2p ops arguments to parser.
    """
    parser.add_argument(
        "--config", default=[], help="Path to configuration .ini file, including filename."
    )
    parser.add_argument("--data", default=[], help="print version number.")
    ops0 = SortConfig()
    for k in ops0.get_all().keys():
        v = dict(default=ops0[k], help="{0} : {1}".format(k, ops0[k]))
        if k in ["fast_disk", "save_folder", "save_path0"]:
            v["default"] = None
            v["type"] = str
        if (type(v["default"]) in [np.ndarray, list]) and len(v["default"]):
            v["nargs"] = "+"
            v["type"] = type(v["default"][0])
        parser.add_argument("--" + k, **v)
    return parser


def parse_args(parser: argparse.ArgumentParser):
    """
    Parses arguments and returns ops with parameters filled in.
    """
    args = parser.parse_args()
    dargs = vars(args)
    ops0 = SortConfig()
    ops = np.load(args.ops, allow_pickle=True).item() if args.ops else {}
    set_param_msg = "->> Setting {0} to {1}"
    # options defined in the cli take precedence over the ones in the ops file
    for k in ops0.get_all():
        default_key = ops0[k]
        args_key = dargs[k]
        if k in ["fast_disk", "save_folder", "save_path0"]:
            if args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))
        elif type(default_key) in [np.ndarray, list]:
            n = np.array(args_key)
            if np.any(n != np.array(default_key)):
                ops[k] = n.astype(type(default_key))
                print(set_param_msg.format(k, ops[k]))
        elif isinstance(default_key, bool):
            args_key = bool(int(args_key))  # bool("0") is true, must convert to int
            if default_key != args_key:
                ops[k] = args_key
                print(set_param_msg.format(k, ops[k]))
        # checks default param to args param by converting args to same type
        elif not (default_key == type(default_key)(args_key)):
            ops[k] = type(default_key)(args_key)
            print(set_param_msg.format(k, ops[k]))
    return args, ops


def main():
    run()


if __name__ == "__main__":
    main()
