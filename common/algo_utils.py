import os
import re
from collections import defaultdict
from multiprocessing import Process, Manager

from subprocess import check_output
from threading import Thread

from typing import Optional, Tuple, Dict

import numpy as np


class Res:
    """ Auxiliary class to mimic pymoo format """

    def __init__(self, X: np.ndarray, F: np.ndarray, history_x: Optional[np.ndarray] = None,
                 history_f: Optional[np.ndarray] = None):
        """

        Args:
            X: best points (pareto front if multi-objective)
            F: function values (shape: (n_points, n_obj_functions)
            history_x: all
        """
        self.X = X
        self.F = F
        self.history_x = history_x
        self.history_f = history_f


def get_history_values_from_res(res: Res) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return an array

    Args:
        res: pymoo result object having `history` field

    Returns:
        X: array of inputs (-1, action_space_size)
        Y: array of obj values (-1, 2)
    """
    X = res.history_x
    Y = res.history_f
    assert Y.ndim == 2, Y.ndim
    assert X.shape == (Y.shape[0], X.shape[-1]), (Y.shape[0], X.shape[-1])
    return X, Y


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.logical_or(np.any(costs[is_efficient] < c, axis=1),
                                                       np.all(costs[is_efficient] == costs[i],
                                                              axis=1))  # Keep any point with a lower cost
    return is_efficient


def pareto_score(pareto_front: np.ndarray):
    """ Compute the score associated to a pareto front (for a 2-objective minimisation task)
    Args:
        pareto_front: np.ndarray of shape (n, 2) containing the 2 objectives at the n points on the pareto front

    Returns:
         score: area under pareto front
    """
    assert np.all(pareto_front > 0)
    assert pareto_front.ndim == 2 and pareto_front.shape[1] == 2, pareto_front.shape

    inds = pareto_front[:, 0].argsort()
    aux = pareto_front[inds]
    i = 0
    x = np.array([0, *aux[:, 0]])
    y = np.array([aux[0, 1], *aux[:, 1]])
    return np.trapz(y, x=x, axis=0)


def aig_stats(design_file, imap_binary, stats):
    imap_command = "read_aiger -f" + design_file + "; print_stats -t"+stats
    try:
        proc = check_output([imap_binary, '-c', imap_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'Stats of AIG' in line:
                ob = re.search(r'pis *= *[0-9]+ */ *[0-9]+', line)
                stats['pis'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'pos *= *[0-9]+', line)
                stats['pos'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'area *= *[0-9]+', line)
                stats['area'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'depth *= *[0-9]+', line)
                stats['depth'] = int(ob.group().split('=')[1].strip())
    except Exception as e:
        print(e)
        return None

    return stats


def extract_features(design_file, yosys_binary='yosys', imap_binary='yosys-abc') -> Dict[str, float]:
    """
    Returns features of a given circuit as a tuple.
    Features are listed below
    """

    try:
        manager = Manager()
        stats = manager.dict()
        p2 = Process(target=aig_stats, args=(design_file, imap_binary, stats))
        p2.start()
        p2.join()
    except AssertionError:

        stats = {}
        thread = Thread(target=aig_stats, args=(
            design_file, imap_binary, stats))

        # thread.daemon = True
        thread.start()
        thread.join()

    # normalized features
    features = defaultdict(float)
    features['pis'] = stats['pis']
    features['pos'] = stats['pos']
    features['area'] = stats['area']
    features['depth'] = stats['depth']
    return features


def get_design_name(design_filepath: str) -> str:
    return os.path.basename(design_filepath).split('.')[0]


class StateDesign:
    """ Data class whose fields are main characteristics of designs"""

    def __init__(self, pis: float, pos: float, area: float, depth: float,
                 obj_1: float, obj_2: float):
        self.pis = pis
        self.pos = pos
        self.area = area
        self.depth = depth
        self.obj_1 = obj_1
        self.obj_2 = obj_2

    def __repr__(self):
        s = 'State:\n'
        s += f'\t- pis: {self.pis}'
        s += f'\t- pos: {self.pos}'
        s += f'\t- area: {self.area}'
        s += f'\t- depth: {self.depth}'
        s += f'\t- obj_1: {self.obj_1}'
        s += f'\t- obj_2: {self.obj_2}'
        return s
