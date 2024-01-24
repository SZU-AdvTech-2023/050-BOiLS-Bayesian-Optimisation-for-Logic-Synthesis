# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved. Redistribution and use in source and binary 
# forms, with or without modification, are permitted provided that the following conditions are met: 
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer. 
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
# following disclaimer in the documentation and/or other materials provided with the distribution. 
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote 
# products derived from this software without specific prior written permission. 
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

from subprocess import check_output
from typing import List, Dict
import re

class Action:
    """ abc action """

    def __init__(self, act_id: str, act_str: str):
        """

        Args:
            act_id: action id
            act_str: string used to apply the action
        """
        self.act_id = act_id
        self.act_str = act_str

    def __repr__(self):
        return f"{self.act_id} -> {self.act_str}"


class ActionSimple(Action):
    """ Action for which act_str = act_id + ;  """

    def __init__(self, act_id: str):
        """

        Args:
            act_id: action id
        """
        super().__init__(act_id=act_id, act_str=act_id + ';')


class ActionCompo(Action):

    def __init__(self, act_id: str):
        """

        Args:
            act_id: action id
        """
        act_str = f' {act_id}; '
        super().__init__(act_id=act_id, act_str=act_str)


BALANCE = ActionSimple('balance')
REWRITE = ActionSimple('rewrite')
REWRITE_Z = ActionSimple('rewrite -z')
REFACTOR = ActionSimple('refactor')
REFACTOR_Z = ActionSimple('refactor -z')
LUT_OPT = ActionSimple('lut_opt')
MAP_FPGA = ActionCompo('map_fpga')


STD_ACTION_SPACE: List[Action] = [
    REWRITE,
    REWRITE_Z,
    REFACTOR,
    REFACTOR_Z,
    BALANCE,
    LUT_OPT
]

EXTENDED_ACTION_SPACE: List[Action] = [
    REWRITE,
    REWRITE_Z,
    REFACTOR,
    REFACTOR_Z,
    BALANCE,
    LUT_OPT,
]


ACTION_SPACES: Dict[str, List[Action]] = {
    'standard': STD_ACTION_SPACE,
    'extended': EXTENDED_ACTION_SPACE,

}

if __name__ == '__main__':
    from pathlib import Path
    import os

    ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
    print(ROOT_PROJECT)
    # sys.path[0] = ROOT_PROJECT
    # from fpga_session import FPGASession

    # for action_space in (EXTENDED_ACTION_SPACE, STD_ACTION_SPACE):
    design_file = os.path.join(ROOT_PROJECT, 'imap_cases/arbiter.aig')
    imap_binary = '/home/eda230218/gitcode/iMAP/bin/imap'

    imap_command = f' read_aiger -f {design_file}; '
    imap_command += 'print_stats -t 0; '
    for action in STD_ACTION_SPACE:
        imap_command += action.act_str
    imap_command += 'map_fpga; print_stats -t 1;'
    print(imap_command)
    proc = check_output([imap_binary, '-c', imap_command])
    line = proc.decode("utf-8").split('\n')[-2].split(':')[-1].strip()
    print(line)
    ob = re.search(r'area *= *[0-9]+', line)
    area = int(ob.group().split('=')[1].strip())
    print(area)
    ob = re.search(r'depth *= *[0-9]+', line)
    depth = int(ob.group().split('=')[1].strip())
