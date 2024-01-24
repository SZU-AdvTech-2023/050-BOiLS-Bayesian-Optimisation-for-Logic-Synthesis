import argparse
import os

from utils.utils_save import ROOT_PROJECT


def add_common_args(parser: argparse.ArgumentParser):
    eda_group = parser.add_argument_group("EDA optimisation")
    eda_group.add_argument("--design_file", type=str, help="Design filepath")
    eda_group.add_argument("--designs_group_id", type=str,
                           required=True, help="ID of group of designs to consider")
    eda_group.add_argument("--frac_part", type=str, default=None,
                           help="Which part of the group to consider (should follow the pattern `i/j`)")
    eda_group.add_argument("--seq_length", type=int, required=True,
                           help="length of the optimal sequence to find")
    eda_group.add_argument("--action_space_id", type=str, default='standard',
                           help="id of action space defining avaible imap optimisation operations")
    eda_group.add_argument("--libary", type=str, default=os.path.join(ROOT_PROJECT, 'lib/imap.cpython-37m-x86_64-linux-gnu.so'),
                           help="library file for mapping")
    eda_group.add_argument("--imap_binary", type=str,
                           default=os.path.join(ROOT_PROJECT, 'imap'))
    eda_group.add_argument("--ref_imap_seq", type=str, default='resyn2',
                           help="sequence of operations to apply to initial design to get reference performance")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing experiment')

    return parser
