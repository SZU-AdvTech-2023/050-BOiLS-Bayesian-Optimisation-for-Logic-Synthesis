# 2021.11.10-add support to new actions
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent)  # should points to the root of the project
sys.path[0] = ROOT_PROJECT

from utils.utils_cmd import parse_list, parse_dict
from typing import List, Union, Tuple, Dict, Any, Optional
import time

def fpga_evaluate(design_file: str, sequence: List[Union[str, int]], imap_binary:str,
                  compute_init_stats: bool = False, verbose: bool = False) \
        -> Tuple[int, int, Dict[str, Any]]:
    """
         Get property of the design after applying sequence of operations `sequence`

        Args:
            design_file: path to the design 'path/to/design.blif'
            sequence: sequence of operations
                        -> either identified by id number
                            0: rewrite
                            1: rewrite -z...
                        -> or by operation names
                            `rewrite`
                            `rewrite -z`
                            ...
            compute_init_stats: whether to compute and store initial stats on delay and area
            verbose: verbosity level
        Returns:
            lut_k, level and extra_info (execution time, initial stats)
        Exception: CalledProcessError
    """
    assert not compute_init_stats
    t_ref = time.time()
    extra_info: Dict[str, Any] = {}
    imap_command = f'read_aiger -f ' + design_file + ';'
    if sequence is None:
        sequence = []
    for action in sequence:
        imap_command+=action+'; '
    imap_command += 'map_fpga; print_stats -t 1;'
    cmd_elements = [imap_binary, '-c', imap_command]
    proc = subprocess.check_output(cmd_elements)
    # read results and extract information
    line = proc.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
    print(line)
    ob = re.search(r'area *= *[0-9]+', line)
    if ob is None:
        print("----" * 10)
        print(f'Command: {" ".join(cmd_elements)}')
        print(f"Out line: {line}")
        print(f"Design: {design_file}")
        print(f"Sequence: {sequence}")
        print("----" * 10)
    area = int(ob.group().split('=')[1].strip())

    ob = re.search(r'depth *= *[0-9]+', line)
    depth = int(ob.group().split('=')[1].strip())

    extra_info['exec_time'] = time.time() - t_ref
    return area, depth, extra_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', list, parse_list)
    parser.add_argument('--design_file', type=str, default='/home/eda230218/gitcode/iMAP/ai_infra/results/data/cavlc.aig',help='path to blif design')
    parser.add_argument('--actions', type=list, help='Sequence of actions')
    parser.add_argument('--imap_binary', type=str, default='/home/eda230218/gitcode/iMAP/bin/imap',help='Sequence of actions')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    results_ = fpga_evaluate(
        design_file=args.design_file,
        sequence=args.actions,
        imap_binary=args.imap_binary,
        verbose=args.verbose
    )
    print(results_)
