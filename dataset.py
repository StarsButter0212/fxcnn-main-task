import os
from typing import Optional, Callable, Dict, List
import yaml
import torch
from torch.utils.data import Dataset
from entry import Entry

filedir = os.path.dirname(os.path.realpath(__file__))

################# dataset #################

class DFTDataset(Dataset):
    def __init__(self, fpath: Optional[str] = None):
        if fpath is None:
            fpath = os.path.join(filedir, "train_dsets", "dft_dataset1.yaml")

        with open(fpath, "r") as f:
            self.obj = [Entry.create(a) for a in yaml.safe_load(f)]

    def __len__(self) -> int:
        return len(self.obj)

    def __getitem__(self, i: int) -> Entry:
        return self.obj[i]

    def get_indices(self, filtfcn: Callable[[Dict], bool]) -> List[int]:
        # return the id of the datasets that passes the filter function
        return [i for (i, obj) in enumerate(self.obj) if filtfcn(obj)]

    def select_indices(self, index: List, type: str) -> List[int]:
        # return the id of the datasets that passes the filter function
        return [i for i in index if self.obj[i]['type'] == type]
