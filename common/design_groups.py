import os
from typing import List, Dict

import numpy as np

from utils.utils_save import get_storage_data_root

DATA_PATH = os.path.join(get_storage_data_root(), 'aig')

EPFL_ARITHMETIC = ['hyp', 'div', 'log2', 'multiplier', 'sqrt', 'square', 'sin', 'bar', 'adder', 'dec','max','arbiter','i2c','cavlc','router','mem_ctrl','ctrl']

DESIGN_GROUPS: Dict[str, List[str]] = {
    'epfl_arithmetic': EPFL_ARITHMETIC,
}

for design in EPFL_ARITHMETIC:
    DESIGN_GROUPS[design] = [design]

AUX_TEST_GP = ['adder', 'bar']
AUX_TEST_ABC_GRAPH = ['adder', 'sin']

DESIGN_GROUPS['aux_test_designs_group'] = AUX_TEST_GP
DESIGN_GROUPS['aux_test_abc_graph'] = AUX_TEST_ABC_GRAPH


def get_designs_path(designs_id: str, frac_part: str = None) -> List[str]:
    """ Get list of filepaths to designs """

    designs_filepath: List[str] = []
    # for design_id in DESIGN_GROUPS[designs_id]:
    designs_filepath.append(os.path.join(DATA_PATH, f'{designs_id}.aig'))
    if frac_part is None:
        s = slice(0, len(designs_filepath))
    else:
        i, j = map(int, frac_part.split('/'))
        assert j > 0 and i > 0, (i, j)
        step = int(np.ceil(len(designs_filepath) / j))
        s = slice((i - 1) * step, i * step)

    return designs_filepath[s]


if __name__ == '__main__':

    designs_id_ = 'test_designs_group'
    N = 6
    for n in range(1, N + 1):
        frac = f'{n}/{N}'
        print(f'{frac} -----> ', end='')
        print(get_designs_path(designs_id=designs_id_, frac_part=frac))
