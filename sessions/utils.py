# 2021.11.10-updated the metrics retrieved from the stats
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import re
from subprocess import check_output
from typing import List, Optional, Tuple, Dict, Union, Any

from utils.utils_misc import log
from sessions.utils_eval import fpga_evaluate


def get_metrics(stats) -> Dict[str, Union[float, int]]:
    """
    parse LUT count and levels from the stats command of imap
    """
    line = stats.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
    results = {}
    ob = re.search(r'pis *= *[0-9]+', line)
    results['pis'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'pos *= *[0-9]+', line)
    results['pos'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'area *= *[0-9]+', line)
    results['area'] = int(ob.group().split('=')[1].strip())

    ob = re.search(r'depth *= *[0-9]+', line)
    results['depth'] = int(ob.group().split('=')[1].strip())

    return results


def get_fpga_design_prop( design_file: str, imap_binary: str, 
                         sequence: List[str] = None, 
                         verbose: Optional[int] = 0) -> Tuple[int, int]:
    """
    Compute and return lut_k and levels associated to a specific design

    Args:
        libary: standard cell library mapping
        design_file: path to the design file
        imap_binary: abc binary path
        sequence: sequence of operations (containing final ';') to apply to the design
        verbose: verbosity level
        write_unmap_design_path: path where to store the design obtained after the sequence of actions have been applied

    Returns:
        lut_K, levels
    """
    imap_command = 'read_aiger -f ' + {design_file} + ';'
    imap_command += ' '.join(sequence)
    imap_command += 'map_fpga; print_stats -t 1;'
    print(f"{imap_binary} -c '{imap_command}'")
    if verbose:
        log(imap_command)
    proc = check_output([imap_binary, '-c', imap_command])
    results = get_metrics(proc)
    print(results['area'], results['depth'])
    return results['area'], results['depth']


def get_design_prop(seq: List[str], design_file: str, 
                    imap_binary: str,  compute_init_stats: bool, verbose: bool = False,
                    ) -> Tuple[int, int, Dict[str, Any]]:
    """
     Get property of the design after applying sequence of operations

    Args:
        seq: sequence of operations
        design_file: path to the design
        libary: library file (asap7.lib)
        verbose: verbosity level
        compute_init_stats: whether to compute and store initial stats
        write_unmap_design_path: path where to store the design obtained after the sequence of actions have been applied

    Returns:
        either:
            - for fpga: lut_k, level
            - for scl: area, delay
    """

        # lut_k, levels = get_fpga_design_prop(
        #     libary=libary,
        #     design_file=design_file,
        #     imap_binary=imap_binary,
        #     sequence=seq,
        #     verbose=verbose,
        #     write_unmap_design_path=write_unmap_design_path
        # )
        # assert not write_unmap_design_path, "[Deprecated] Does not support this option anymore"
    area, depth, extra_info = fpga_evaluate(design_file=design_file, sequence=seq, imap_binary=imap_binary,
                                                compute_init_stats=compute_init_stats, verbose=verbose,
                                               )
    return area, depth, extra_info
