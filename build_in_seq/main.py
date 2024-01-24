import abc
import os
from typing import List, Tuple, Dict, Union, Type, Any

from sessions.utils import get_design_prop
from utils.utils_save import get_storage_root, load_w_pickle, save_w_pickle


class BuildInSeq(abc.ABC):
    sequence: List[str]

    def __init__(self, libary: str, design_file: str, imap_binary: str):
        self.libary = libary
        self.design_file = design_file
        self.imap_binary = imap_binary

    def fpga(self, compute_init_stats: bool,  verbose: bool = False) \
            -> Tuple[int, int, Dict[str, Any]]:
        """ Return lut-6 and levels after application of predifined sequence """
        return get_design_prop(
            design_file=self.design_file,
            imap_binary=self.imap_binary,
            seq=self.sequence,
            compute_init_stats=compute_init_stats,
            verbose=verbose,
        )

    @staticmethod
    def seq_length() -> int:
        raise NotImplementedError()


class Resyn(BuildInSeq):
    sequence = [
        'balance',
        'rewrite',
        'rewrite -z',
        'balance',
        'rewrite -z',
        'balance'
    ]

    def __init__(self, libary: str, design_file: str, imap_binary: str):
        """
            balance; rewrite; rewrite -z; balance; rewrite -z; balance
        """

        super().__init__(libary, design_file, imap_binary)

    @staticmethod
    def seq_length() -> int:
        return len(Resyn.sequence)


resyn2_seq = [
    'balance',
    'rewrite',
    'refactor',
    'balance',
    'rewrite',
    'rewrite -z',
    'balance',
    'refactor -z',
    'rewrite -z',
    'balance'
]


class Resyn2(BuildInSeq):
    sequence = resyn2_seq

    def __init__(self, libary: str, design_file: str, imap_binary: str):
        """
            balance; rewrite; refactor; balance; rewrite; rewrite –z; balance; refactor –z; rewrite –z; balance;
        """

        super().__init__(libary, design_file, imap_binary)

    @staticmethod
    def seq_length() -> int:
        return len(Resyn2.sequence)


class InitDesign(BuildInSeq):
    sequence = []

    def __init__(self, libary: str, design_file: str, imap_binary: str):
        """
            No action, evaluate initial design
        """

        super().__init__(libary, design_file, imap_binary)

    @staticmethod
    def seq_length() -> int:
        return len(InitDesign.sequence)


BUILD_IN_SEQ: Dict[str, Union[Type[InitDesign], Type[Resyn], Type[Resyn2]]] = dict(
    init=InitDesign,
    resyn=Resyn,
    resyn2=Resyn2
)


class RefObj:

    def __init__(self, design_file: str, imap_binary: str, libary: str,
                 ref_imap_seq: str):
        """
            Args:
                design_file: path to the design
                libary: library file (asap7.lib)
                imap_binary: (probably yosys-abc)
                ref_imap_seq: sequence of operations to apply to initial design to get reference performance
            """
        self.design_file = design_file
        self.imap_binary = imap_binary
        self.libary = libary
        self.ref_imap_seq = ref_imap_seq

        self.design_name = os.path.basename(design_file).split('.')[0]

    def get_config(self) -> Dict[str, Any]:
        return dict(
            design_file=self.design_file,
            design_name=self.design_name,
            ref_imap_seq=self.ref_imap_seq
        )

    def ref_path(self) -> str:
        return os.path.join(get_storage_root(), 'refs', self.ref_imap_seq, self.design_name)

    def get_refs(self) -> Tuple[float, float]:
        if os.path.exists(os.path.join(self.ref_path(), 'refs.pkl')):
            refs = load_w_pickle(self.ref_path(), 'refs.pkl')
            return refs['ref_1'], refs['ref_2']

        biseq_cl = BUILD_IN_SEQ[self.ref_imap_seq]
        biseq = biseq_cl(libary=self.libary, design_file=self.design_file,
                         imap_binary=self.imap_binary)

        ref_1, ref_2, extra_info = biseq.fpga( verbose=True, compute_init_stats=False,
                                                  )
        os.makedirs(self.ref_path(), exist_ok=True)
        ref_obj = dict(ref_1=ref_1, ref_2=ref_2, config=self.get_config(), exec_time=extra_info['exec_time'])
        save_w_pickle(ref_obj, self.ref_path(), 'refs.pkl')
        return ref_1, ref_2


if __name__ == '__main__':
    print('; '.join(resyn2_seq))
