from typing import Optional
from GA.nsga2_exp import NSGA2Exp
from common.argparse import add_common_args
import traceback

import argparse
import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT


def main(designs_group_id: str, seq_length: int, action_space_id: str,
         libary: str,
         imap_binary: str,
         pop_size: int, seed: int, n_gen: int, eta_mutation: float,
         eta_cross: int, prob_cross: float, selection: str,
         overwrite: bool, n_parallel: int,
         ref_imap_seq: str, verbose: bool = True):
    """
    Args:
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        action_space_id: id of action space defining available abc optimisation operations
        n_parallel: number of threads to compute the refs
        libary: library file (asap7.lib)
        imap_binary: (probably yosys-abc)
        ref_imap_seq: sequence of operations to apply to initial design to get reference performance
        pop_size: population size for SGA
        n_gen: number of generations
        eta_mutation: eta parameter for int_pm mutation
        eta_cross: eta parameter for crossover
        prob_cross: prob parameter for crossover
        selection: selection process
        seed: reproducibility seed
        overwrite: Overwrite existing experiment
    """
    exp: NSGA2Exp = NSGA2Exp(
        designs_group_id=designs_group_id,
        seq_length=seq_length,
        action_space_id=action_space_id,
        libary=libary,
        imap_binary=imap_binary,
        n_gen=n_gen,
        eta_mutation=eta_mutation,
        eta_cross=eta_cross,
        prob_cross=prob_cross,
        selection=selection,
        seed=seed,
        pop_size=pop_size,
        ref_imap_seq=ref_imap_seq,
        n_parallel=n_parallel
    )
    exist = exp.exists()
    if exist and not overwrite:
        exp.log(f"Experiment already trained: stored in {exp.exp_path()}")
        return exp.exp_path()
    elif exist:
        exp.log(f"Overwrite experiment: {exp.exp_path()}")
    result_dir = exp.exp_path()
    exp.log(f'result dir: {result_dir}')
    os.makedirs(result_dir, exist_ok=True)
    logs = ''
    exc: Optional[Exception] = None
    try:
        res = exp.run(verbose=verbose)
        exp.save_results(res)
    except Exception as e:
        logs = traceback.format_exc()
        exc = e
    f = open(os.path.join(result_dir, 'logs.txt'), "a")
    f.write(logs)
    f.close()
    if exc is not None:
        raise exc
    return exp.exp_path()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Performs logic synthesis optimization using NSAG2')
    parser = add_common_args(parser)
    parser.add_argument("--n_parallel", type=int, default=1,
                        help="number of threads to compute the stats")

    # SGA Search
    parser.add_argument("--pop_size", type=int, default=8,
                        help="population size for SGA")
    parser.add_argument("--n_gen", type=int, required=True,
                        help="Number of generations")
    parser.add_argument("--eta_cross", type=float, default=15,
                        help="eta parameter for int_sbx crossover")
    parser.add_argument("--eta_mute", type=float, default=20,
                        help="eta parameter for int_pm mutation")
    parser.add_argument("--prob_cross", type=float,
                        default=0.9, help="Probability of crossover")
    parser.add_argument("--selection", type=str, default="random", choices=('random', 'tournament'),
                        help="Selection process.")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for reproducibility")

    args_ = parser.parse_args()

    if not os.path.isabs(args_.libary):
        args_.libary = os.path.join(ROOT_PROJECT, args_.libary)

    main(
        designs_group_id=args_.designs_group_id,
        seq_length=args_.seq_length,
        action_space_id=args_.action_space_id,
        libary=args_.libary,
        imap_binary=args_.imap_binary,
        seed=args_.seed,
        ref_imap_seq=args_.ref_imap_seq,
        pop_size=args_.pop_size,
        n_gen=args_.n_gen,
        eta_cross=args_.eta_cross,
        eta_mutation=args_.eta_mute,
        prob_cross=args_.prob_cross,
        selection=args_.selection,
        overwrite=args_.overwrite,
        n_parallel=args_.n_parallel,
    )
