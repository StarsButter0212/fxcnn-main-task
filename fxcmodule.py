import re
import torch
import torch.nn as nn
import argparse
import warnings
from typing import Dict, List, Union, Tuple
from dqc.api.getxc import get_xc
from xcmodels import HybridXC
from evaluator import XCDNNEvaluator, PySCFEvaluator


from tqdm import tqdm
from datetime import datetime
from utils import accuracy, AverageMeter
from user_model import Shared_Bottom, MixtureOfExperts
from entry import Entry, System


###################### training module ######################
class FXCNN(nn.Module):
    def __init__(self, hparams: Dict, entries: List[Dict] = [],
                 device: torch.device = torch.device('cpu')):
        # hparams contains ():
        # * libxc: str
        # * nhid: int
        # * ndepths: int
        # * nn_with_skip: bool
        # * ninpmode: int
        # * sinpmode: int
        # * outmultmode: int
        # * iew: float
        # * aew: float
        # * dmw: float
        # * densw: float
        super().__init__()

        nnxcmode = hparams.get("nnxcmode", None)
        if nnxcmode is not None:
            warnings.warn("--nnxcmode flag is deprecated, please use --ninpmode and --outmultmode")
        if nnxcmode is None:
            pass
        elif nnxcmode == 1:
            hparams["ninpmode"] = 1
            hparams["outmultmode"] = 1
        elif nnxcmode == 2:
            hparams["ninpmode"] = 2
            hparams["outmultmode"] = 2
        elif nnxcmode == 3:
            hparams["ninpmode"] = 2
            hparams["outmultmode"] = 1
        else:
            raise RuntimeError("Invalid value of nnxcmode: %s" % str(nnxcmode))

        self.device = device
        self.evl = self._construct_model(hparams, entries)
        self._hparams = hparams

    def _construct_model(self, hparams: Dict, entries: List[Dict] = []):

        # set the weights
        weights = {
            "ie": hparams.get("iew", 1340.),
            "ae": hparams.get("aew", 1340.),
            "dens": hparams.get("densw", 170.),
            "dm": hparams.get("dmw", 220.),
        }
        # set arbitrarily, but more weights on the energy as they are the
        # ones we know from experiments (not from simulations)
        dweights = {
            "ie": 1.0,
            "ae": 1.0,
            "dens": 1.0,
            "dm": 1.0,
        }
        if hparams["weight_loss"]:
            self.weights = weights
        else:
            self.weights = hparams['weight_set']
        self.type_indices = {x: i for i, x in enumerate(self.weights.keys())}
        self.use_pyscf = hparams.get("pyscf", False)

        print('self.weights:', self.weights)
        print()

        if not self.use_pyscf:
            if not hparams["purennmode"]:
                # prepare the nn xc model
                libxc_dqc = hparams["libxc"].replace(",", "+")
                family = get_xc(libxc_dqc).family
                if family == 1:             # LDA
                    ninp = 2                # input numbers
                elif family == 2:           # GGA
                    ninp = 3
                else:
                    raise RuntimeError("Unimplemented nn for xc family %d" % family)
            else:
                ninp = 1

            # setup the xc nn model
            if hparams["user_model"] and not hparams["multi_tasks"]:
                print("# of using user's model.")
                ninp = 75
                hidden_expert = 128
                output_size = 75
                num_experts = 10
                self.nnmodel = MixtureOfExperts(ninp, hidden_expert, output_size,     # MoE simple
                                                num_experts).to(torch.double)
            elif hparams["user_model"] and hparams["multi_tasks"]:
                print("# of using user's task model.")
                nhid, ndepths, towers_hidden = 32, 2, 64
                self.nnmodel = Shared_Bottom(ninp, nhid, ndepths, towers_hidden).to(torch.double)

            elif hparams.get("nneq", None) is None: # Equation of the neural network
                print("# of using orignal model.")
                nhid = hparams["nhid"]
                ndepths = hparams["ndepths"]
                nn_with_skip = hparams.get("nn_with_skip", False)
                modeltype = hparams.get("modeltype", 1)
                self.nnmodel = construct_nn_model(ninp, nhid, ndepths, nn_with_skip,
                                                  modeltype).to(torch.double)

            xc_nnxc = HybridXC(hparams["libxc"], self.nnmodel,
                               ninpmode=hparams["ninpmode"],
                               sinpmode=hparams.get("sinpmode", 1),
                               aweight0=hparams.get("nnweight0", 0.0),
                               bweight0=hparams.get("xcweight0", 1.0),
                               outmultmode=hparams["outmultmode"],
                               purennmode=hparams["purennmode"],
                               device=self.device)

            always_attach = hparams.get("always_attach", False)
            grid, truncate = hparams["grid_config"].split('_')
            CCSD_dm0 = hparams.get("CCSD_dm0", False)
            warning_display = hparams.get("warning_display", False)

            return XCDNNEvaluator(xc_nnxc, self.weights, always_attach=always_attach,
                                  entries=entries, warning_display=warning_display,
                                  grid=grid, truncate=truncate,
                                  CCSD_dm0=CCSD_dm0, device=self.device)
        else:
            # if using pyscf, no neural network is constructed
            # dummy parameter required just to make it run without error
            self.dummy_param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))
            return PySCFEvaluator(hparams["libxc"], weights, device=self.device)

    def forward(self, batch: Dict, task: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        losses, deviation = 0, 0
        if task:
            self.evl.xc.upgrad_nnxc(task)

        if isinstance(batch, list):
            for x in batch:
                loss, val, true_val = self.evl.calc_loss_function(x)
                deviation += self.evl.calc_deviation(x, val, true_val)
                losses += loss
            loss = losses / len(batch)
            deviation = deviation / len(batch)
        else:   # batch_size = 1
            loss, val, true_val = self.evl.calc_loss_function(batch)
            deviation = self.evl.calc_deviation(batch, val, true_val)

        if self.use_pyscf:
            loss = loss + self.dummy_param * 0

        return loss, deviation

    @staticmethod
    def get_trainer_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # params that are specific to the model
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # arguments to be stored in the hparams file
        # model hyperparams
        parser.add_argument("--nhid", type=int, default=32,
                            help="The number of elements in hidden layers")
        parser.add_argument("--ndepths", type=int, default=3,
                            help="The number of hidden layers depths")
        parser.add_argument("--nn_with_skip", action="store_const", const=True, default=False,
                            help="Add skip connection in the neural network")
        parser.add_argument("--modeltype", type=int, default=1,
                            help="The neural network model type")
        parser.add_argument("--nneq", type=str, default=None,
                            help=("Equation of the neural network. If specified, then other nn "
                                  "architecture arguments are ignored (nhid, ndepths, nn_with_skip)."))
        parser.add_argument("--libxc", type=str, default="lda_x",   # "gga_x_pbe,gga_c_pbe
                            help="Initial xc to be used")
        parser.add_argument("--nnweight0", type=float, default=0.0,
                            help="Initial weight of the nn in hybrid xc-nn")
        parser.add_argument("--xcweight0", type=float, default=1.0,
                            help="Initial weight of the xc in hybrid xc-nn")
        parser.add_argument("--ninpmode", type=int, default=3,
                            help="The mode to decide the transformation of density to the NN input")
        parser.add_argument("--sinpmode", type=int, default=3,
                            help="The mode to decide the transformation of normalized grad density to the NN input")
        parser.add_argument("--outmultmode", type=int, default=1,
                            help="The mode to decide the Eks from NN output")
        parser.add_argument("--nnxcmode", type=int,
                            help="The mode to decide how to compute Exc from NN output (deprecated, do not use)")
        parser.add_argument("--pyscf", action="store_const", default=False, const=True,
                            help="Using pyscf calculation. If activated, the nn-related arguments are ignored.")

        # hparams for the loss function
        parser.add_argument("--iew", type=float, default=1340.0,
                            help="Weight of ionization energy")
        parser.add_argument("--aew", type=float, default=1340.0,
                            help="Weight of atomization energy")
        parser.add_argument("--dmw", type=float, default=220.0,
                            help="Weight of density matrix")
        parser.add_argument("--densw", type=float, default=170.0,
                            help="Weight of density profile loss")

        # hparams for optimizer
        parser.add_argument('--use_OneCycleLR', type=bool, default=True,
                            help="Learning rate scheduler")
        parser.add_argument('--alpha', '-a', type=float, default=0.12)
        # ----------------------------------------------------------------------------------------------
        parser.add_argument("--optimizer", type=str, default="adam",
                            help="Optimizer algorithm")
        parser.add_argument("--wdecay", type=float, default=0.0,
                            help="Weight decay of the algorithm (i.e. L2 regularization)")
        parser.add_argument("--split_opt", action="store_const", default=False, const=True,
                            help="Flag to split optimizer based on the dataset type")
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate for all types data")
        parser.add_argument("--ielr", type=float, default=1e-4,
                            help="Learning rate for ionization energy (chosen if there is --split_opt)")
        parser.add_argument("--aelr", type=float, default=1e-4,
                            help="Learning rate for atomization energy (ignored if no --split_opt)")
        parser.add_argument("--dmlr", type=float, default=1e-4,
                            help="Learning rate for density matrix (ignored if no --split_opt)")
        parser.add_argument("--denslr", type=float, default=1e-3,
                            help="Learning rate for density profile (ignored if no --split_opt)")
        return parser


class NNModel(nn.Module):
    def __init__(self, ninp: int, nhid: int, ndepths: int, with_skip: bool = False):
        super().__init__()
        layers = []
        activations = []
        if with_skip:
            skip_weights = []
            conn_weights = []

        for i in range(ndepths):
            n1 = ninp if i == 0 else nhid
            layers.append(torch.nn.Linear(n1, nhid))
            activations.append(torch.nn.Softplus())
            if with_skip and i >= 1:
                # using Linear instead of parameter to avoid userwarning
                # of parameterlist not supporting set attributes
                conn_weights.append(torch.nn.Linear(1, 1, bias=False))
                skip_weights.append(torch.nn.Linear(1, 1, bias=False))

        layers.append(torch.nn.Linear(nhid, 1, bias=False))

        # construct the nn parameters
        self.layers = torch.nn.ModuleList(layers)
        self.activations = torch.nn.ModuleList(activations)
        self.with_skip = with_skip
        if with_skip:
            self.conn_weights = torch.nn.ModuleList(conn_weights)
            self.skip_weights = torch.nn.ModuleList(skip_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            y = self.layers[i](x)

            # activation (no activation at the last layer)
            if i < len(self.activations):
                y = self.activations[i](y)

            # skip connection (no skip at the first and last layer)
            if self.with_skip and i >= 1 and i < len(self.layers) - 1:
                y1 = self.conn_weights[i - 1](y.unsqueeze(-1))
                y2 = self.skip_weights[i - 1](x.unsqueeze(-1))
                y = (y1 + y2).squeeze(-1)
            x = y
        return x

class ExpM1Activation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - 1

def construct_nn_model(ninp: int, nhid: int, ndepths: int, with_skip: bool = False, modeltype: int = 1):
    # construct the neural network model of the xc energy
    if not with_skip:
        if modeltype == 1:
            # old version, to enable loading the old models
            layers = []
            for i in range(ndepths):
                n1 = ninp if i == 0 else nhid
                layers.append(torch.nn.Linear(n1, nhid))
                layers.append(torch.nn.Softplus())
            layers.append(torch.nn.Linear(nhid, 1, bias=False))
            return torch.nn.Sequential(*layers)
        elif modeltype == 2:
            layers = []
            for i in range(ndepths):
                n1 = ninp if i == 0 else nhid
                layers.append(torch.nn.Linear(n1, nhid))
                if i < ndepths - 1:
                    layers.append(torch.nn.Softplus())
                else:
                    layers.append(ExpM1Activation())
            layers.append(torch.nn.Linear(nhid, 1, bias=False))
            return torch.nn.Sequential(*layers)
    else:
        return NNModel(ninp, nhid, ndepths, with_skip)

class LinearOutputNet(nn.Module):
    def __init__(self, nhid):
        super(LinearOutputNet, self).__init__()
        self.fc = nn.Linear(nhid, 1, bias=False)

    def forward(self, x):
        out = self.fc(x.float())
        return out
