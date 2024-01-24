import matplotlib.pyplot as plt
import numpy as np
import os
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Any, Type

from build_in_seq.main import BUILD_IN_SEQ, RefObj
from common.action_space import Action, ACTION_SPACES
from common.design_groups import get_designs_path
from sessions.utils import get_design_prop
from utils.utils_plot import get_cummin, plot_mean_std
from utils.utils_save import get_storage_root, load_w_pickle, save_w_pickle


# DEPRECATED
class EDAExp(ABC):
    color = None
    linestyle = None

    def __init__(self, design_file: str, seq_length: int, action_space_id: str,
                 libary: str,
                 imap_binary: str,
                 ref_imap_seq: Optional[str] = None):
        """
        Args:
            design_file: path to the design
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            libary: library file (asap7.lib)
            imap_binary: (probably yosys-abc)
            ref_imap_seq: sequence of operations to apply to initial design to get reference performance
        """
        assert os.path.exists(libary), os.path.abspath(libary)
        self.libary = libary
        self.imap_binary = imap_binary

        self.exec_time = 0
        self.design_file = design_file
        self.design_name = os.path.basename(design_file).split('.')[0]
        self.seq_length = seq_length

        self.action_space_id = action_space_id

        if ref_imap_seq is None:
            ref_imap_seq = 'init'  # evaluate initial design
        self.ref_imap_seq = ref_imap_seq
        biseq_cl = BUILD_IN_SEQ[ref_imap_seq]
        self.biseq = biseq_cl(libary=self.libary, design_file=self.design_file,
                              imap_binary=self.imap_binary)

        ref_obj = RefObj(design_file=self.design_file,  imap_binary=self.imap_binary,
                         libary=self.libary, ref_imap_seq=self.ref_imap_seq)

        self.ref_1, self.ref_2 = ref_obj.get_refs()

        self.action_space = self.get_action_space_()

    @abstractmethod
    def exists(self) -> bool:
        """ Check if experiment already exists """
        raise NotImplementedError()

    def get_action_space_(self) -> List[Action]:
        return self.get_action_space(action_space_id=self.action_space_id)

    @staticmethod
    def get_action_space(action_space_id: str) -> List[Action]:
        assert action_space_id in ACTION_SPACES, (action_space_id, list(ACTION_SPACES.keys()))
        return ACTION_SPACES[action_space_id]

    @abstractmethod
    def exp_id(self) -> str:
        raise NotImplementedError()

    def get_prop(self, seq: List[int], compute_init_stats: bool = False) -> Tuple[float, float, Dict[str, Any]]:
        sequence = [self.action_space[i].act_id for i in seq]
        return get_design_prop(seq=sequence, design_file=self.design_file, 
                                imap_binary=self.imap_binary,
                               compute_init_stats=compute_init_stats)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        return dict(
            design_file=self.design_file,
            design_name=self.design_name,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            ref_imap_seq=self.ref_imap_seq
        )

    @property
    @abstractmethod
    def meta_method_id(self) -> str:
        """ Id for the meta method (will appear in the result-path) """
        raise NotImplementedError()

    def exp_path(self) -> str:
        return self.get_exp_path_aux(
            meta_method_id=self.meta_method_id,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_name=self.design_name,
            ref_imap_seq=self.ref_imap_seq
        )

    @staticmethod
    def get_exp_path_aux(meta_method_id: str, seq_length: int, action_space_id: str,
                         exp_id: str, design_name: str, ref_imap_seq: str = None) -> str:
        aux = f"{seq_length}_act-{action_space_id}"
        if ref_imap_seq != 'resyn2':
            aux += f"_{ref_imap_seq}"
        return os.path.join(get_storage_root(), meta_method_id, aux, exp_id, design_name)

    @property
    def obj1_id(self):
        return 'area'

    @property
    def obj2_id(self):
        return 'depth'

    @property
    def action_space_length(self):
        return len(self.action_space)

    @staticmethod
    def plot_regret_qor(qors: np.ndarray, add_ref: bool = False, ax: Optional[Axes] = None,
                        exp_cls: Optional[Type['EDAExp']] = None,
                        **plot_kw) -> Axes:
        """
        Plot regret QoR curve

        Args:
            qors: array of qors obtained using some algorithm
            add_ref: whether to add initial QoR of 2 (QoR of the ref)
            ax: axis
            exp_cls: subclass of EDAExp used to get these results
            **plot_kw: plot kwargs

        Returns:
            ax: the axis
        """
        if ax is None:
            ax = plt.subplot()
        if 'c' not in plot_kw and 'color' not in plot_kw:
            plot_kw['c'] = exp_cls.color
        if 'linestyle' not in plot_kw:
            plot_kw['linestyle'] = exp_cls.linestyle
        qors = np.atleast_2d(qors)
        if add_ref:
            aux_qors = []
            for qor in qors:
                aux_qors = np.concatenate([np.array([2]), qor])
            qors = np.array(aux_qors)
        regret_qors = get_cummin(qors)
        ax = plot_mean_std(regret_qors, ax=ax, **plot_kw)
        return ax


class EADExp:

    def __init__(self, designs_group_id: str, seq_length: int,action_space_id: str,
                 libary: str,
                 imap_binary: str, 
                 n_parallel: int = 1,
                 ref_imap_seq: Optional[str] = None):
        """
        Args:
            designs_group_id: id of the designs group
            seq_length: length of the optimal sequence to find
            mapping: either scl of fpga mapping
            use_yosys: whether to use yosys-abc or abc_py
            action_space_id: id of action space defining available abc optimisation operations
            libary: library file (asap7.lib)
            imap_binary: (probably yosys-abc)
            n_parallel: number of threads to compute the refs
            ref_imap_seq: sequence of operations to apply to initial design to get reference performance
        """
        assert os.path.exists(libary), os.path.abspath(libary)
        self.libary = libary
        self.imap_binary = imap_binary

        self.exec_time = 0
        self.designs_group_id = designs_group_id
        self.design_files = get_designs_path(self.designs_group_id)
        self.design_names = list(
            map(lambda design_path: os.path.basename(design_path).split('.')[0], self.design_files))
        self.seq_length = seq_length

        self.action_space_id = action_space_id

        if ref_imap_seq is None:
            ref_imap_seq = 'init'  # evaluate initial design
        self.ref_imap_seq = ref_imap_seq

        biseq_cl = BUILD_IN_SEQ[ref_imap_seq]
        self.biseq = biseq_cl(libary=self.libary, design_file=self.design_files[0],
                              imap_binary=self.imap_binary)

        self.refs_1: List[float] = []
        self.refs_2: List[float] = []

        refs = Parallel(n_jobs=n_parallel, backend="multiprocessing")(delayed(self.get_ref)(self.design_files[ind])
                                                                      for ind in tqdm(range(len(self.design_files))))

        for refs_1_2 in refs:
            self.refs_1.append(refs_1_2[0])
            self.refs_2.append(refs_1_2[1])

        self.action_space: List[Action] = self.get_action_space()

    @abstractmethod
    def exists(self) -> bool:
        """ Check if experiment already exists """
        raise NotImplementedError()

    def get_ref(self, design_file: str) -> Tuple[float, float]:
        """ Return either area and delay or lut and levels """

        ref_obj = RefObj(design_file=design_file, imap_binary=self.imap_binary,
                         libary=self.libary,  ref_imap_seq=self.ref_imap_seq,
                         )

        ref_1, ref_2 = ref_obj.get_refs()

        return ref_1, ref_2

    def get_action_space(self) -> List[Action]:
        assert self.action_space_id in ACTION_SPACES, (self.action_space_id, list(ACTION_SPACES.keys()))
        return ACTION_SPACES[self.action_space_id]

    @abstractmethod
    def exp_id(self) -> str:
        raise NotImplementedError()

    # def get_prop(self, seq: List[int]) -> Tuple[float, float]:
    #     sequence = [self.action_space[i].act_str for i in seq]
    #     return get_design_prop(seq=sequence, design_file=self.design_file, mapping=self.mapping,
    #                             imap_binary=self.imap_binary, )

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        return dict(
            design_files_id=self.designs_group_id,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            ref_imap_seq=self.ref_imap_seq,
        )

    @property
    @abstractmethod
    def meta_method_id(self) -> str:
        """ Id for the meta method (will appear in the result-path) """
        raise NotImplementedError()

    def exp_path(self) -> str:
        return self.get_exp_path_aux(
            meta_method_id=self.meta_method_id,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_files_id=self.designs_group_id,
            ref_imap_seq=self.ref_imap_seq
        )

    @staticmethod
    def get_exp_path_aux(meta_method_id: str,  seq_length: int, action_space_id: str,
                         exp_id: str, design_files_id: str, ref_imap_seq: str) -> str:
        return os.path.join(get_storage_root(), meta_method_id,
                            f"_seq-{seq_length}_ref-{ref_imap_seq}"
                            f"_act-{action_space_id}",
                            exp_id,
                            design_files_id)

    @staticmethod
    def get_eval_ckpt_root_path(action_space_id: str) -> str:
        return os.path.join(get_storage_root(),
                            f"_{action_space_id}")

    @property
    def eval_ckpt_root_path(self) -> str:
        return self.get_eval_ckpt_root_path(action_space_id=self.action_space_id)


    @property
    def action_space_length(self):
        return len(self.action_space)


class seqEADExp(EADExp):

    def __init__(self, designs_group_id: str, seq_length: int, n_universal_seqs: int,
                 action_space_id: str, libary: str, imap_binary: str, n_parallel: int = 1, 
                 ref_imap_seq: Optional[str] = None):
        """
        Looking for `n_universal_seqs` universal sequences working for all circuits
        Args:
            designs_group_id: id of the designs group
            seq_length: length of the optimal sequence to find
            n_universal_seqs: number of sequences
            mapping: either scl of fpga mapping
            action_space_id: id of action space defining available abc optimisation operations
            libary: library file (asap7.lib)
            imap_binary: (probably yosys-abc)
            n_parallel: number of threads to compute the refs
            ref_imap_seq: sequence of operations to apply to initial design to get reference performance
        """
        super().__init__(
            designs_group_id=designs_group_id,
            seq_length=seq_length,
            action_space_id=action_space_id,
            libary=libary,
            imap_binary=imap_binary,
            n_parallel=n_parallel,
            ref_imap_seq=ref_imap_seq
        )
        self.n_universal_seqs = n_universal_seqs

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config['n_universal_seqs'] = self.n_universal_seqs
        return config

    def exp_path(self) -> str:
        return self.get_exp_path_aux(
            meta_method_id=self.meta_method_id,
            seq_length=self.seq_length,
            action_space_id=self.action_space_id,
            exp_id=self.exp_id(),
            design_files_id=self.designs_group_id,
            ref_imap_seq=self.ref_imap_seq,
            n_universal_seqs=self.n_universal_seqs
        )

    @staticmethod
    def get_exp_path_aux(meta_method_id: str, seq_length: int, action_space_id: str,
                         exp_id: str, design_files_id: str, ref_imap_seq: str, n_universal_seqs: int) -> str:
        return os.path.join(get_storage_root(), meta_method_id,
                            f"_seq-{seq_length}_ref-{ref_imap_seq}"
                            f"_act-{action_space_id}_n-univesal-{n_universal_seqs}",
                            exp_id,
                            design_files_id)


class Checkpoint:
    """
    Useful class for checkpointing (store the inputs tested so far and the ratios associated to first and second
        objectives for each input
     """

    def __init__(self, samples: np.ndarray, full_objs_1: np.ndarray, full_objs_2: np.ndarray):
        self.samples = samples
        self.full_objs_1 = full_objs_1
        self.full_objs_2 = full_objs_2
