import os
import csv
import sys
import time
import copy
import math
import argparse
import timeit
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional, List, Union, Tuple

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])

from dataset import DFTDataset
from fxcmodule import FXCNN
from utils import hashstr
from entry import Entry, System
from dqc.utils.datastruct import SpinParam

class XCDNNEvaluator(torch.nn.Module):
    """
    Kohn-Sham model where the XC functional is replaced by a neural network
    """
    def __init__(self, entries: List[Dict] = [],
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        self.device = device

        # register system-specific buffer
        self._init_dm_buffer(entries)

    def _dm0_buffer_name(self, obj) -> str:
        # returns the buffer name
        return "dm0_" + hashstr(str(obj))

    def _init_dm_buffer(self, entries: List[Dict]) -> None:
        # initialize the dm0 cache for each system in the entries as buffer
        for entry_dct in entries:
            entry = Entry.create(entry_dct, device=self.device)
            systems = entry.get_systems()
            for syst in systems:
                buffer_name = self._dm0_buffer_name(syst)
                dqc_syst = syst.get_dqc_system()
                dqc_hamilt = dqc_syst.get_hamiltonian()
                dqc_hamilt.build()
                nao = dqc_hamilt.nao
                if dqc_syst.spin != 0:
                    shape = (2, nao, nao)
                else:
                    shape = (nao, nao)
                val = torch.zeros(shape, dtype=torch.double)
                self.register_buffer(buffer_name, val)

    def _get_dm0_buffer(self, system: System) -> \
        Tuple[Union[None, torch.Tensor, SpinParam[torch.Tensor]], Optional[str]]:
        # get the dm0 cache from the buffer

        # Returns a tuple of the dm0 which is a tensor if it has been written or
        # None if no dm0 has been stored before
        # and the buffer name if the buffer is created during initialization or
        # None otherwise
        buffer_name = self._dm0_buffer_name(system)
        dm0: Optional[torch.Tensor] = getattr(self, buffer_name, None)

        buffer_exists = dm0 is not None
        buffer_written = buffer_exists and torch.any(dm0 != 0.0)
        if not buffer_written:
            dm0_res: Union[None, torch.Tensor, SpinParam[torch.Tensor]] = None
        elif system.get_dqc_system().spin != 0:
            dm0_res = SpinParam(u=dm0[0].detach(), d=dm0[1].detach())
        else:
            dm0_res = dm0

        return dm0_res, (buffer_name if buffer_exists else None)

def get_program_argparse() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()

    # training hyperparams
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--device", type=str, default='CPU',
                        help="train batch size.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="train batch size.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of the processes.")
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--file", type=str, default='./test_dsets/dens_gauss2.yaml',
                        help="The training dataset file")
    parser.add_argument("--exclude_types", type=str, nargs="*", default=[],
                        help="Exclude several types of dataset")
    return parser

def get_datasets(hparams: Dict):

    num_workers = hparams["num_workers"]
    pin_memory = hparams["pin_memory"]

    # load the dataset and split into train and val
    dset = DFTDataset(fpath=hparams.get("file", None))

    general_filter = lambda obj: obj["type"] not in hparams["exclude_types"]
    all_idxs = dset.get_indices(general_filter)
    print('# of dataset samples: ', len(all_idxs))

    dset_all = Subset(dset, all_idxs)
    dloader_all = DataLoader(dset_all, batch_size=None, num_workers=num_workers,
                             pin_memory=pin_memory, shuffle=False)

    return dloader_all


def main(hparams: Dict):

    """GPU or CPU."""
    if hparams["device"] == 'GPU':
        device = torch.device('cuda')
        print('The code uses a GPU.')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU.')
    print('-'*50)

    dloader_all = get_datasets(hparams)

    # optional step to inform the model what entries are so that it can
    # prepare specific buffer for each entries / systems
    entries = []
    for dct in dloader_all:
        entries.append(dct)

    # XCDNNEvaluator(entries=entries, device=device)

    for index, entry_raw in enumerate(entries):
        # get the entry object
        entry = Entry.create(entry_raw, device=device)
        name = entry['name'].split('of')[-1]
        true_val = entry.get_true_val()

        print(name)
        plt.plot(true_val.detach().cpu().numpy(), label='true_val')
        plt.title(name)
        plt.legend(loc='best')
        plt.show()
        print('--> {:0>3d} of {} true_val is: {}'.format(index+1, len(entries), true_val.shape))

    print('# finished!')



if __name__ == '__main__':

    # parsing the hyperparams
    parser = get_program_argparse()
    args = parser.parse_args()

    # putting all the hyperparameters in a dictionary
    hparams = vars(args)

    # main
    main(hparams)