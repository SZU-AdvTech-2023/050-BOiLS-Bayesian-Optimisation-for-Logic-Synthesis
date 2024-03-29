import traceback
from multiprocessing import set_start_method

import argparse
import os
import sys
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

from common.action_space import ACTION_SPACES
from sentence_transformers import SentenceTransformer
from Boils.utils import InputTransformation, SentenceBertInputTransform
from utils.utils_save import load_w_pickle, get_storage_models_root
from common.argparse import add_common_args
from Boils.boils_exp import BoilsExp
from typing import Optional


def main(designs_group_id: str, seq_length: int, action_space_id: str,
         libary: str,
         imap_binary: str,
         seed: int, n_initial: int, standardise: bool, ard: bool,
         acq: str,
         n_total_evals: int,
         overwrite: bool, n_parallel: int,
         ref_imap_seq: str, objective: str, device: Optional[int], length_init_discrete_factor: float, failtol: float,
         input_transformation: Optional[InputTransformation] = None, kernel_type: Optional[str] = None,
         verbose: bool = True):
    """
    Args:
        designs_group_id: id of the designs group
        seq_length: length of the optimal sequence to find
        mapping: either scl of fpga mapping
        action_space_id: id of action space defining available abc optimisation operations
        n_parallel: number of threads to compute the refs
        libary: library file (asap7.lib)
        imap_binary: (probably yosys-abc)
        lut_inputs: number of LUT inputs (2 < num < 33)
        ref_imap_seq: sequence of operations to apply to initial design to get reference performance
        n_total_evals: number of sequences to evaluate
        n_initial: number of initial points to test before building first surrogate model
        standardise: whether to standardise the ys for the GP fit
        ard: whether to enable automatic relevance determination
        acq: choice of the acquisition function.
        seed: reproducibility seed
        overwrite: Overwrite existing experiment
        objective: which objective to optimize ('lut', 'level', 'both')
        device: gpu device id
        kernel_type: type of kernel to use for the surrogate model of BOiLS (default is `transformed_overlap`)
        failtol: failure tolerance factor for BOiLS
        length_init_discrete_factor: length_init_discrete=seq_length * length_init_discrete_factor
    """
    if device >= 0:
        set_start_method('spawn')
    exp: BoilsExp = BoilsExp(
        designs_group_id=designs_group_id,
        seq_length=seq_length,
        action_space_id=action_space_id,
        libary=libary,
        imap_binary=imap_binary,
        seed=seed,
        n_initial=n_initial,
        standardise=standardise,
        ard=ard,
        acq=acq,
        input_transformation=input_transformation,
        kernel_type=kernel_type,
        failtol=failtol,
        length_init_discrete_factor=length_init_discrete_factor,
        ref_imap_seq=ref_imap_seq,
        n_parallel=n_parallel,
        objective=objective,
        overwrite=overwrite
    )
    exist = exp.exists()
    if exist and not overwrite:
        if not os.path.exists(os.path.join(exp.exp_path(), 'ckpt.pkl')):
            exist = False
        else:
            # check if enough sequences have been evaluated
            ckpt = load_w_pickle(exp.exp_path(), 'ckpt.pkl')
            if len(ckpt.full_objs_1) < n_total_evals:
                exist = False
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
        res = exp.run(n_total_evals=n_total_evals, verbose=verbose, device=device, overwrite=overwrite)
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
                                     description='Performs logic synthesis optimization using BOiLS')
    parser = add_common_args(parser)
    parser.add_argument("--n_parallel", type=int, default=1, help="number of threads to compute the stats")

    # BOiLS Search
    # parser.add_argument("--n_total_evals", type=int, required=True, help="number of sequences to evaluate")
    parser.add_argument("--n_total_evals", type=int, default=25, help="number of sequences to evaluate")
    parser.add_argument("--n_initial", type=int, default=20, help="Number of initial random points to evaluate")
    parser.add_argument("--standardise", action='store_true', help="whether to standardise the ys for the GP fit")
    parser.add_argument("--ard", action='store_true',default=' ', help="whether to use ard")
    parser.add_argument("--failtol", type=float,default=40, help="BOiLS failure tolerance")
    parser.add_argument("--length_init_discrete_factor", type=float,default=.666,
                        help="length_init_discrete=seq_length * length_init_discrete_factor")
    parser.add_argument("--acq", type=str,
                        choices=('ei', 'ucb'),
                        default='ei',
                        help="acquistion function")
    parser.add_argument("--objective", type=str,default='both', choices=('lut', 'both', 'level', 'min_improvements'),
                        help="which objective should be optimized")
    parser.add_argument("--kernel_type", type=str,
                        choices=('transformed_overlap', 's-bert-matern52', 's-bert-rbf', 'ssk'),
                        default='transformed_overlap', help="type of kernel to use for the surrogate model of BOiLS "
                                           "(default is `transformed_overlap`)")
    parser.add_argument("--embedder_path", type=str, default=None, help="Dimensionality of embedding")
    parser.add_argument("--name_embed", action='store_true', help="Whether to use operators'names")
    parser.add_argument("--seed", type=int, nargs='+', help="seed for reproducibility",default=[0,1,2])
    parser.add_argument("--device", type=int, help="cuda id (cpu if None or negative)")

    args_ = parser.parse_args()

    if not os.path.isabs(args_.libary):
        args_.libary = os.path.join(ROOT_PROJECT, args_.libary)

    input_transformation_ = None
    if args_.kernel_type == 's-bert-matern52':
        kernel_type_ = 's-bert-matern52'
    elif args_.kernel_type == 's-bert-rbf':
        kernel_type_ = 's-bert-rbf'
    elif args_.kernel_type == 'transformed_overlap':
        kernel_type_ = 'transformed_overlap'
    elif args_.kernel_type == 'ssk':
        kernel_type_ = 'ssk'
    elif args_.kernel_type is None:
        kernel_type_ = None
    else:
        raise ValueError()
    if 's-bert' in kernel_type_:
        if os.path.isabs(args_.embedder_path):
            assert hasattr(args_, 'embedder_path'), "to use sentence-BERT embedder one needs to specify the path " \
                                                    "to the trained model"
            embedder_path = args_.embedder_path
        else:
            embedder_path = os.path.join(get_storage_models_root(), args_.embedder_path)
        sbert_model = SentenceTransformer(model_name_or_path=embedder_path)
        name_embedder = None
        if args_.name_embed:
            def aux_name_embedder(ind_: int) -> str:
                assert int(ind_) == ind_, ind_
                return ACTION_SPACES[args_.action_space_id][int(ind_)].act_id


            name_embedder = aux_name_embedder
        input_transformation_ = SentenceBertInputTransform(
            sbert_model=sbert_model, id=f'sbert-{sbert_model.get_sentence_embedding_dimension()}d',
            name_embed=args_.name_embed,
            name_embedder=name_embedder
        )
    for seed in args_.seed:
        main(
            designs_group_id=args_.designs_group_id,
            seq_length=args_.seq_length,
            action_space_id=args_.action_space_id,
            libary=args_.libary,
            imap_binary=args_.imap_binary,
            seed=seed,
            ref_imap_seq=args_.ref_imap_seq,
            n_total_evals=args_.n_total_evals,
            standardise=args_.standardise,
            acq=args_.acq,
            ard=args_.ard,
            device=args_.device,
            n_initial=args_.n_initial,
            overwrite=args_.overwrite,
            n_parallel=args_.n_parallel,
            objective=args_.objective,
            input_transformation=input_transformation_,
            kernel_type=kernel_type_,
            failtol=args_.failtol,
            length_init_discrete_factor=args_.length_init_discrete_factor,
        )
