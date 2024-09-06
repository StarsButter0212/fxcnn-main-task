import os
import argparse
import itertools
from typing import List, Dict, Optional, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from fxcnn.fxcmodule import FXCNN
from fxcnn.dataset import DFTDataset
from fxcnn.train import get_program_argparse, get_datasets

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

def get_infer_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--dataset", type=str, default='test_dsets/ae_mols.yaml',
                        help="The dataset where the inference is taking place")
    parser.add_argument("--chkpts", type=str, default=[],
                        nargs="+",
                        help="Checkpoints where the models are loaded from")
    parser.add_argument("--writeto", type=str, default='outputs/test_dsets.txt',
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



if __name__ == "__main__":

    parser = get_program_argparse()
    parser = FXCNN.get_trainer_argparse(parser)
    parser = get_infer_argparse(parser)
    parser = Plotter.get_plot_argparse(parser)
    args = parser.parse_args()

    # putting all the hyperparameters in a dictionary
    hparams = vars(args)
    startline = hparams["startline"]
    writer = Writer(hparams["writeto"], startline).open()

    dloader_train, dloader_val = get_datasets(hparams)
    print('-' * 50)

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

    # load the model and the dataset
    models = []
    if startline is None:
        writer.write("# Checkpoints: %s" % ("|".join(hparams["chkpts"])))
    for chkpt in hparams["chkpts"]:
        # if chkpt is a file, the load the checkpoint
        if os.path.exists(chkpt):
            device = torch.device('cuda')
            mdl = FXCNN(hparams, entries, device).to(device)
            mdl.load_state_dict(torch.load(chkpt, map_location=device))
            if hparams["showparams"]:
                print("Parameters for %s:" % chkpt)
                print(list(mdl.parameters()))
        # otherwise, it is assumed as libxc string for pyscf
        else:
            mhparams = {
                "libxc": chkpt,
                "pyscf": True,
            }
            device = torch.device('cpu')
            hparams["libxc"] = chkpt.split('/')[2]
            print('# of chkpt:', chkpt.split('/')[2])
            hparams["pyscf"] = True
            mdl = FXCNN(hparams, entries, device)
        models.append(mdl)
    dset = DFTDataset(hparams["dataset"])
    print('# of test samples: ', len(dset))
    print('-' * 50)

    # calculate the losses for all entries and models
    all_losses = []
    istart = 1 if startline is None else startline
    maxentries = hparams.get("maxentries", None) or (len(dset) + 100)
    entries = 0
    for i in range(len(dset)):
        if i + 1 < istart:
            continue
        if entries >= maxentries:
            break
        losses = []
        for model in models:
            loss, val, true_val = model.evl.calc_loss_function(dset[i])
            losses.append(float(model.evl.calc_deviation(dset[i], val, true_val).item()))
        losses_str = list2str(losses)
        writer.write("%d out of %d: %s: (%s)" % (i + 1, len(dset), dset[i]["name"], losses_str))
        all_losses.append(losses)
        entries += 1

    # get the mean
    all_losses = np.array(all_losses)
    writer.write("     Mean absolute error (MAE): %s" % list2str(np.mean(np.abs(all_losses), axis=0)))
    writer.write("Root mean squared error (RMSE): %s" % list2str(np.sqrt(np.mean(all_losses ** 2, axis=0))))
    writer.close()

    # show the plot
    if hparams["plot"]:
        plotter = Plotter(len(hparams["chkpts"]), hparams, all_losses)
        plotter.show()
