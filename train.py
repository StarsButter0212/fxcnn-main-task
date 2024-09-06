import os
import csv
import sys
import time
import copy
import math
import torch
import shutil
import itertools
import argparse
import timeit
import numpy as np

from datetime import datetime
from tqdm import tqdm
from ray import tune
from typing import Dict, Optional
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])

from dataset import DFTDataset
from fxcmodule import FXCNN
from user_model import Trainer, Tester
from utils import get_exp_version
from ray.tune.search.hyperopt import HyperOptSearch


import warnings
warnings.filterwarnings("ignore")

######################## hparams part ########################
def get_program_argparse() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_const", default=True, const=True,
                        help="Flag to record the progress")
    parser.add_argument("--version", type=str, default='lda_x',            # lda_x, pbe_xc
                        help="The training version, if exists, then resume the training")
    parser.add_argument("--logdir", type=str, default="./outputs/",
                        help="The log directory relative to this file's path")
    parser.add_argument("--seed", type=int, default=321, help="Random seed")

    # hparams not used for the actual training
    # (only for different execution modes of this file)
    parser.add_argument("--cmd", action="store_const", default=False, const=True,
                        help="Run the training via command line")
    parser.add_argument("--tune", action="store_const", default=False, const=True,
                        help="Run the hyperparameters tuning")

    # training hyperparams
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--batch_size", type=int, default=3,
                        help="train batch size.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="The number of the processes.")
    parser.add_argument("--pin_memory", type=bool, default=True)

    parser.add_argument("--grid_config", type=str, default='sg3_dasgupta',
                        help="Construct the radial grid transformation.")
    parser.add_argument('--user_model', type=bool, default=True,
                        help="Using the user's model")
    parser.add_argument("--weight_loss", type=bool, default=False,
                        help="Flag to if using weight to compute loss")
    parser.add_argument("--weight_set", type=float, nargs="*", default=[1.0, 1.0, 1.0],
                        help="Set weights to compute loss")

    parser.add_argument("--multi_tasks", type=bool, default=True,
                        help="Use multi-tasks algorithm for computing loss weight")
    parser.add_argument("--tasks", type=str, nargs="*", default=['ae', 'ie', 'dens'],
                        help="Types of dataset for training neural network")
    parser.add_argument('--algorithm', type=str, default='none',
                        help="multi-tasks using algorithm.")
    parser.add_argument("--normalization_type", type=str, default="loss+",
                        help="Grad normalization_type")
    parser.add_argument("--CCSD_dm0", type=bool, default=False,
                        help="The dm0 initialization mode.")
    parser.add_argument("--purennmode", type=bool, default=False,
                        help="The mode to decide the if use pure xcnn to caculate edensityxc")
    parser.add_argument("--warning_display", action="store_const", default=True, const=True,
                        help="If display warning message in the training epochs")

    # 即使迭代不收敛，也始终传播梯度
    parser.add_argument("--always_attach", action="store_const", default=False, const=True,
                        help="Always propagate gradient even if the iteration does not converge")
    # -----------------------------------------------------------------------------------------------
    parser.add_argument("--clipval", type=float, default=0,
                        help="Clip gradients with norm above this value. 0 means no clipping.")
    parser.add_argument("--max_epochs", type=int, default=300,
                        help="Maximum number of epochs")
    parser.add_argument("--trainingfile", type=str, default=None,
                        help="The training dataset file")
    parser.add_argument("--tvset", type=int, default=2,
                        help="Training/validation set")
    parser.add_argument("--exclude_types", type=str, nargs="*", default=[],
                        help="Exclude several types of dataset")
    parser.add_argument("--tiny_dset", action="store_const", default=False, const=True,
                        help="Flag to use tiny dataset for sanity check")
    return parser

def convert_to_tune_config(hparams: Dict) -> Dict:

    # set the hyperparameters to be tuned
    res = copy.deepcopy(hparams)
    split_opt = hparams["split_opt"]
    exclude_types = hparams["exclude_types"]
    res["record"] = True  # if hparams are tuned, it must be recorded

    res["nhid"] = tune.choice([16, 32, 64])
    res["ndepths"] = tune.choice([1, 2, 3, 4])
    res["ninpmode"] = tune.choice([1, 2, 3])
    res["outmultmode"] = tune.choice([1, 2])
    if (split_opt and "ie" not in exclude_types) or (not split_opt):
        res["ielr"] = tune.loguniform(1e-5, 3e-3)
    if split_opt and "ae" not in exclude_types:
        res["aelr"] = tune.loguniform(1e-5, 3e-3)
    if split_opt and "dm" not in exclude_types:
        res["dmlr"] = tune.loguniform(1e-5, 3e-3)
    if split_opt and "dens" not in exclude_types:
        res["denslr"] = tune.loguniform(1e-5, 3e-3)
    return res

######################## dataset and training part ########################
def lambda_xs(xs):
    return list(xs)

# load the datasets and returns the dataloader for training and validation
def get_datasets(hparams: Dict):
    dloader_train, dloader_val = {}, {}

    from utils import subs_present, get_atoms
    batch_size = hparams["batch_size"]
    num_workers = hparams["num_workers"]
    pin_memory = hparams["pin_memory"]

    # load the dataset and split into train and val
    dset = DFTDataset(fpath = hparams.get("trainingfile", None))
    tvset = hparams["tvset"]

    if tvset == 1:
        # train_atoms = ["H", "He", "Li", "Be", "B", "C"]
        val_atoms = ["N", "O", "F", "Ne"]
    elif tvset == 2:
        # train_atoms = ["H", "Li", "B", "C", "O", "Ne"]
        val_atoms = ["He", "Be", "N", "F", "P", "S"]

    general_filter = lambda obj: obj["type"] not in hparams["exclude_types"]
    all_idxs = dset.get_indices(general_filter)
    print('# of dataset samples: ', len(all_idxs))

    val_filter = lambda obj: (subs_present(val_atoms, get_atoms(obj["name"].split()[-1]))
                              and general_filter(obj))
    val_idxs = dset.get_indices(val_filter)
    train_idxs = list(set(all_idxs) - set(val_idxs))

    if hparams["tiny_dset"]:
        val_idxs = val_idxs[:1]
        train_idxs = train_idxs[:1]

    if hparams["user_model"] and hparams["multi_tasks"]:
        train_idxs_ie = dset.select_indices(train_idxs, 'ie')
        train_idxs_ae = dset.select_indices(train_idxs, 'ae')
        train_idxs_dens = dset.select_indices(train_idxs, 'dens')

        train_idxs_ie = np.resize(train_idxs_ie, len(train_idxs_dens))
        train_idxs_ae = np.resize(train_idxs_ae, len(train_idxs_dens))

        dset_train_ie = Subset(dset, train_idxs_ie)
        dset_train_ae = Subset(dset, train_idxs_ae)
        dset_train_dens = Subset(dset, train_idxs_dens)

        dloader_train_ie = DataLoader(dset_train_ie, batch_size=batch_size, num_workers=num_workers,
                                      collate_fn=lambda_xs, pin_memory=pin_memory, shuffle=True)
        dloader_train_ae = DataLoader(dset_train_ae, batch_size=batch_size, num_workers=num_workers,
                                      collate_fn=lambda_xs, pin_memory=pin_memory, shuffle=True)
        dloader_train_dens = DataLoader(dset_train_dens, batch_size=batch_size, num_workers=num_workers,
                                      collate_fn=lambda_xs, pin_memory=pin_memory, shuffle=True)

        dloader_train['ie'], dloader_train['ae'], dloader_train['dens'] = dloader_train_ie, \
                                                                          dloader_train_ae, dloader_train_dens
    else:
        dset_train = Subset(dset, train_idxs)
        dloader_train = DataLoader(dset_train, batch_size=None, num_workers=num_workers,
                                   pin_memory=pin_memory, shuffle=True)

    dset_val = Subset(dset, val_idxs)
    dloader_val = DataLoader(dset_val, batch_size=None, num_workers=num_workers,
                             pin_memory=pin_memory)

    return dloader_train, dloader_val

def run_training(hparams: Dict):
    iterations = hparams['max_epochs']
    tasks = list(set(hparams['tasks']) - set(hparams['exclude_types']))
    tasks.sort(key=hparams['tasks'].index)

    dloader_train, dloader_val = get_datasets(hparams)
    if hparams["multi_tasks"]:
        print('# of training samples: ', len(dloader_train['ae']))
    else:
        print('# of training samples: ', len(dloader_train))
    print('# of test samples: ', len(dloader_val))
    print('-' * 50)

    # optional step to inform the model what entries are so that it can
    # prepare specific buffer for each entries / systems
    entries = []
    if hparams["user_model"] and hparams["multi_tasks"]:
        # for dloader in [dloader_train, dloader_val]:
        #     for task in tasks:
        #         entries += list(itertools.chain.from_iterable(dloader[task]))

        for task in tasks:
            entries += list(itertools.chain.from_iterable(dloader_train[task]))
        for dct in dloader_val:
            entries.append(dct)
    else:
        for dct in itertools.chain(dloader_train, dloader_val):
            entries.append(dct)

    print('Set a FXCNN model.')
    if hparams["user_model"] and hparams["multi_tasks"]:
        print('# of n_tasks:', tasks)
    print('# of n_iterations:', iterations)

    # create the lightning module and the datasets
    model = FXCNN(hparams, entries, device).to(device)

    trainer = Trainer(model, hparams)
    tester = Tester(model, hparams)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 50)

    """Output files."""
    save_time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    if hparams["pyscf"]:
        out_dir = hparams["logdir"] + 'pyscf/' + hparams["version"]
    else:
        out_dir = hparams["logdir"] + 'xcnn'

    os.makedirs(out_dir + '/' + 'csv/', exist_ok=True)
    os.makedirs(out_dir+ '/' + 'csv/' + 'temp/', exist_ok=True)
    weights = ', '.join(map(str, hparams['weight_set'].values()))
    file_name = 'log_' + str(save_time) + '_{' + weights + '}.csv'
    logname = out_dir + '/' + 'csv/' + 'temp/' + file_name

    with open(logname, 'w', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train_loss', 'loss_ae', 'loss_ie', 'loss_dens',
                            'test_loss', 'MAE_ae', 'DEV_ie', 'DEV_dens'])

    os.makedirs(out_dir + '/' + 'model/', exist_ok=True)
    file_model = out_dir + '/' + 'model/' + 'model--' + str(save_time) + '.pth'

    print('Start training of the FXCNN model with CCCBDB dataset.\n'
          'The training result is displayed in this terminal every epoch.\n'
          'The result, prediction, and trained model '
          'are saved in the output directory.\n'
          'Wait for a while...')

    MAE_min = np.inf
    for epoch in range(iterations):
        torch.cuda.empty_cache()
        train_loss, loss_ae, loss_ie, loss_dens = trainer.train(dloader_train, epoch)
        test_loss, MAE_ae, DEV_ie, DEV_dens = tester.test(dloader_val)

        tqdm.write('\33[34m【test_loss】: {0}, 【MAE_ae】: {1}, 【DEV_ie】: {2}, '
                   '【DEV_dens】: {3}\33[0m'.format(test_loss, MAE_ae, DEV_ie, DEV_dens))
        time.sleep(0.1)

        with open(logname, 'a', newline='') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, loss_ae, loss_ie, loss_dens,
                                test_loss, MAE_ae, DEV_ie, DEV_dens])

        # Save the model with the best performance at last 10 epochs
        if epoch >= (iterations - 10):
            if float(MAE_ae + DEV_ie + DEV_dens) < MAE_min:
                MAE_min = float(MAE_ae + DEV_ie + DEV_dens)
                tester.save_model(model, file_model)

    tqdm.write('\33[34m【MAE_min】: {0}\33[0m'.format(MAE_min))
    shutil.move(logname, out_dir + '/' + 'csv/' + file_name)
    print('The training has finished.')


if __name__ == "__main__":

    # parsing the hyperparams
    parser = get_program_argparse()
    parser = FXCNN.get_trainer_argparse(parser)
    args = parser.parse_args()

    # putting all the hyperparameters in a dictionary
    hparams = vars(args)

    if hparams["version"] == 'lda_x':
        wset = [[0.49, 0.11, 0.4]]
    else:
        wset = [[0.05, 0.1, 0.85]]

    for w in wset:
        hparams['weight_set'] = dict(zip(['ae', 'ie', 'dens'], w))

        # set the random seed
        torch.manual_seed(args.seed)

        """GPU or CPU."""
        if torch.cuda.is_available() and not hparams["pyscf"]:
            device = torch.device('cuda')
            print('The code uses a GPU.')
        else:
            device = torch.device('cpu')
            print('The code uses a CPU.')
        print('-'*50)

        bestval = run_training(hparams)
        print("Output:", bestval)
