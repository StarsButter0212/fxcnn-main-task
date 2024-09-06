import os
import pandas as pd
import argparse
import itertools
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from fxcnn.fxcmodule import FXCNN
from fxcnn.dataset import DFTDataset
from fxcnn.train import get_program_argparse, get_datasets


def get_infer_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=['test_dsets/ie_gauss2.yaml', 'test_dsets/dens_gauss2.yaml', 'test_dsets/ae_gauss2.yaml'],
                        help="The dataset where the inference is taking place")
    parser.add_argument("--chkpts", type=str, nargs="+",
                        default='outputs/xcnn/model/model.pth',
                        help="Checkpoints where the models are loaded from")
    parser.add_argument("--writeto", type=str, default='outputs/predict/test_dsets.csv',
                        help="If specified, then write the results into the file")
    parser.add_argument("--startline", type=int, default=None,
                        help="The starting entry (in int) of the dataset file to evaluate")
    parser.add_argument("--maxentries", type=int, default=None,
                        help="The number of entries to be inferred from the dataset file")
    parser.add_argument("--showparams", action="store_const", default=False, const=True,
                        help="If enabled, then show the parameters of loaded checkpoints")

    # plot options
    parser.add_argument("--plot", action="store_const", default=False, const=True,
                        help="If present, plot the values")
    return parser

def list2str(x: List[float], fmt: str = "%.4e", sep: str = ", ") -> str:
    # convert a list of float into a string
    return sep.join([fmt % xx for xx in x])

class Writer(object):
    def __init__(self, writeto: Optional[str], startline: Optional[int]):
        self.writeto = writeto
        self.startline = startline

    def open(self):
        if self.writeto is not None:
            mode = "w" if self.startline is None else "a"
            self.f = open(self.writeto, mode)
        return self

    def write(self, s: str):
        print(s)
        if self.writeto is not None:
            self.f.write(s + "\n")
            self.f.flush()

    def close(self):
        if self.writeto is not None:
            self.f.close()

def get_subsets(entry_names: List[str], values: np.ndarray) -> Tuple[List[str], List[List[str]], List[np.ndarray]]:
    # split the dataset into various subsets, then returns:
    # * list of the subset names (e.g. hydrocarbons)
    # * list of a list of entry names in each subset
    # * list of 2D numpy arrays for each subset
    # NOTE: This function is specific for each dataset, so any changes in the
    # dataset must be reflected in this function

    # Gauss2 atomization energy dataset
    if len(entry_names) == 110:
        subset_names = ["hydrocarbons", "subs-hydrocarbons", "other-1", "other-2"]
        indices = [
            [2, 3, 4, 5, 21, 22, 23, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 107],
            [29, 49, 50, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 108],
            [0, 1, 6, 7, 8, 9, 10, 11, 19, 20, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 53, 57, 61, 64, 66, 67, 68, 69, 70, 105, 109],
            [12, 13, 14, 15, 16, 17, 18, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 65, 106],
        ]
    elif len(entry_names) == 105:
        subset_names = ["hydrocarbons", "subs-hydrocarbons", "other-1", "other-2"]
        indices = [
            [2, 3, 4, 19, 20, 21, 68, 69, 70, 71, 72, 73, 74, 75, 76, 101],
            [27, 47, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 102, 103],
            [0, 1, 5, 6, 7, 8, 9, 10, 17, 18, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 50, 54, 58, 61, 63, 64, 65, 66, 67, 99, 104],
            [11, 12, 13, 14, 15, 16, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 55, 56, 57, 59, 60, 62, 100],
        ]
    # ionization energy
    elif len(entry_names) == 18:
        subset_names = ["trainval", "other"]
        indices = [
            [6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17],
        ]
    else:
        raise RuntimeError("Unknown dataset with %d entries" % len(entry_names))

    # get the subset of the entry names and the values
    svalues = []
    sentrynames = []
    for idx in indices:
        svalues.append(values[np.array(idx)])
        sentrynames.append([entry_names[i] for i in idx])
    return subset_names, sentrynames, svalues

class Plotter(object):
    # this object provides interface to plot the losses
    def __init__(self, ntypes: int, hparams: Dict, all_losses: List[List[float]]):
        self.losses = all_losses  # self.losses[i_entry][i_model]
        self.hparams = self._set_default_hparams(ntypes, hparams)
        self.ntypes = ntypes

    def show(self):
        # show the plot of the losses in the current axes
        assert len(self.losses) > 0
        plt.plot(self.losses, 'o')
        if self.hparams["labels"]:
            plt.legend(self.hparams["labels"])
        if self.hparams["title"]:
            plt.title(self.hparams["title"])
        if self.hparams["xlabel"]:
            plt.xlabel(self.hparams["xlabel"])
        if self.hparams["ylabel"]:
            plt.ylabel(self.hparams["ylabel"])
        plt.show()

    def _set_default_hparams(self, ntypes: int, hparams: Dict):
        # set the default hparams
        # currently there's nothing to do here
        return hparams

    @staticmethod
    def get_plot_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--labels", type=str, nargs="*",
                            help="Labels for the checkpoints")
        parser.add_argument("--title", type=str,
                            help="Title of the plot")
        parser.add_argument("--ylabel", type=str,
                            help="y-axis label of the plot")
        parser.add_argument("--xlabel", type=str,
                            help="x-axis label of the plot")
        return parser



if __name__ == "__main__":

    parser = get_program_argparse()
    parser = FXCNN.get_trainer_argparse(parser)
    parser = get_infer_argparse(parser)
    parser = Plotter.get_plot_argparse(parser)
    args = parser.parse_args()

    # putting all the hyperparameters in a dictionary
    hparams = vars(args)

    """GPU or CPU."""
    if torch.cuda.is_available() and not hparams["pyscf"]:
        device = torch.device('cuda')
        print('The code uses a GPU.')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU.')
    print('-' * 50)

    dloader_train, dloader_val = get_datasets(hparams)

    # optional step to inform the model what entries are so that it can
    # prepare specific buffer for each entries / systems
    tasks = list(set(hparams['tasks']) - set(hparams['exclude_types']))

    entries = []
    if hparams["user_model"] and hparams["multi_tasks"]:
        for task in tasks:
            entries += list(itertools.chain.from_iterable(dloader_train[task]))
        for dct in dloader_val:
            entries.append(dct)
    else:
        for dct in itertools.chain(dloader_train, dloader_val):
            entries.append(dct)

    # load the model
    chkpt = hparams["chkpts"]
    if os.path.exists(chkpt):
        mdl = FXCNN(hparams, entries, device).to(device)
        mdl.load_state_dict(torch.load(chkpt, map_location=device))
        if hparams["showparams"]:
            print("Parameters for %s:" % chkpt)
            print(list(mdl.parameters()))

    # load the dataset
    dsets = []
    for dset in hparams["dataset"]:
        dsets.append(DFTDataset(dset))
    print('# of test samples: ', len(dsets))
    print('-' * 50)

    # calculate the losses for all entries and models
    columns, values = [], []
    with torch.no_grad():
        for file, dset in zip(hparams["dataset"], dsets):
            names, losses = [], []
            task = file.split('/')[-1].split('_')[0]
            mdl.evl.xc.upgrad_nnxc(task)
            for i in range(len(dset)):
                names.append(dset[i]["name"].split(' ')[3])
                _, val, true_val = mdl.evl.calc_loss_function(dset[i])
                losses.append(float(mdl.evl.calc_deviation(dset[i], val, true_val).item()))
            if task == 'ae':
                subset_names, entryname_subsets, val_subsets = get_subsets(names, np.asarray(losses))
                subset_names.insert(0, 'AE 104')
                val_subsets.insert(0, np.asarray(losses))
                columns.extend(subset_names)
                values.extend(val_subsets)
            elif task == 'ie':
                columns.extend('IP 18')
                values.append(losses)
            elif task == 'dens':
                columns.extend('DP 99')
                values.append(losses)

        # get the mean
        MAE = [np.mean(np.abs(val)) for val in values]
        RMSE = [np.sqrt(np.mean(val ** 2)) for val in values]

        # create the excel file for save test results
        save_dict = dict(zip(columns, values))
        df = pd.DataFrame.from_dict(save_dict, orient='index')
        new_df = df.T
        insert_row = max(map(len, values))
        new_df.loc[insert_row + 1] = MAE
        new_df.loc[insert_row + 2] = RMSE
        new_df.to_csv(hparams["writeto"], index=False)

        # show the plot
        if hparams["plot"]:
            plotter = Plotter(len(hparams["chkpts"]), hparams, losses)
            plotter.show()
