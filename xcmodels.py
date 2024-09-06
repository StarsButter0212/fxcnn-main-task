import torch
import torch.nn as nn
import numpy as np
from typing import Union, Iterator, List
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam
from dqc.utils.safeops import safenorm, safepow
from dqc.api.getxc import get_xc
from abc import abstractproperty, abstractmethod

class BaseNNXC(BaseXC, torch.nn.Module):
    @abstractproperty
    def family(self) -> int:
        pass

    @abstractmethod
    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        pass

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        # torch.nn.module prefix has no ending dot, while xt prefix has
        nnprefix = prefix if prefix == "" else prefix[:-1]
        return [name for (name, param) in self.named_parameters(prefix=nnprefix)]

class NNLDA(BaseNNXC):
    # neural network xc functional of LDA (only receives the density as input)

    def __init__(self, nnmodel: torch.nn.Module, ninpmode: int = 1, outmultmode: int = 1,
                 device: torch.device = torch.device("cpu")):
        # nnmodel should receives input with shape (..., 2)
        # where the last dimension is for:
        # (0) total density: (n_up + n_dn), and
        # (1) spin density: (n_up - n_dn) / (n_up + n_dn)
        # the output of the model must have shape of (..., 1)
        # it represents the energy density per density per volume
        super().__init__()
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.outmultmode = outmultmode
        self.device = device

        self.task = None
        self.aux_loss = None
        self.shells = 75  # SG-2 grid params
        self.angular_points = 302

    @property
    def family(self) -> int:
        return 1

    # Returns the xc energy density (energy per unit volume)
    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # collect the total density (n) and the spin density (xi)
        if isinstance(densinfo, ValGrad):           # unpolarized case
            n = densinfo.value.unsqueeze(-1)        # (*BD, nr, 1)
            xi = torch.zeros_like(n)
        else:                                       # polarized case
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd                             # (*BD, nr, 1)
            xi = (nu - nd) / (n + 1e-18)            # avoiding nan, spin density

        # decide how to transform the density to be the input of nn
        ninp = get_n_input(n, self.ninpmode)

        # get the neural network output
        if self.task:
            x = torch.cat((ninp, xi), dim=-1).to(self.device)  # (*BD, nr, 2)
            nnout = self.nnmodel(x, self.task)      # (*BD, nr, nhid)
        else:
            x = torch.cat((ninp, xi), dim=-1).to(self.device)  # (*BD, nr, 2)
            nnout = self.nnmodel(x)                 # (*BD, nr, 1)

        if isinstance(nnout, tuple):
            nnout, self.aux_loss = nnout
        res = get_out_from_nnout(nnout, n, self.outmultmode,
                                 device=self.device)  # (*BD, nr, 1)

        res = res.squeeze(-1)
        return res

class NNGGA(BaseNNXC):
    # neural network xc functional of GGA (receives the density and grad as inputs)

    def __init__(self, nnmodel: torch.nn.Module, ninpmode: int = 1, sinpmode: int = 1, outmultmode: int = 1,
                 device: torch.device = torch.device("cpu")):
        # nnmodel should receives input with shape (..., 3)
        # where the last dimension is for:
        # (0) total density (n): (n_up + n_dn), and
        # (1) spin density (xi): (n_up - n_dn) / (n_up + n_dn)
        # (2) normalized gradients (s): |del(n)| / [2(3*pi^2)^(1/3) * n^(4/3)]
        # the output of the model must have shape of (..., 1)
        # it represents the energy density per density per volume
        super().__init__()
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.sinpmode = sinpmode
        self.outmultmode = outmultmode
        self.device = device

        self.task = None
        self.aux_loss = None

    @property
    def family(self) -> int:
        return 2

    # Returns the xc energy density (energy per unit volume)
    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # densinfo.grad : (*BD, nr, 3)

        # collect the total density (n), spin density (xi), and normalized gradients (s)
        a = 6.187335452560271                       # 2 * (3 * np.pi ** 2) ** (1.0 / 3)
        if isinstance(densinfo, ValGrad):           # unpolarized case
            assert densinfo.grad is not None
            n = densinfo.value.unsqueeze(-1)        # (*BD, nr, 1)
            xi = torch.zeros_like(n)
            s = safenorm(densinfo.grad, dim=-1).unsqueeze(-1)
        else:                                       # polarized case
            assert densinfo.u.grad is not None
            assert densinfo.d.grad is not None
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd                             # (*BD, nr, 1)
            n_offset = n + 1e-18                    # avoiding nan
            xi = (nu - nd) / n_offset
            s = safenorm(densinfo.u.grad + densinfo.d.grad, dim=-1).unsqueeze(-1)

        # normalize the gradient
        if self.sinpmode // 10 == 0:
            s = s / a * safepow(n, -4.0 / 3)

        # decide how to transform the density to be the input of nn
        ninp = get_n_input(n, self.ninpmode)
        sinp = get_n_input(s, self.sinpmode % 10)

        # get the neural network output
        x = torch.cat((ninp, xi, sinp), dim=-1).to(self.device)     # (*BD, nr, 3)
        if self.task:
            nnout = self.nnmodel(x, self.task)          # (*BD, nr, 1)
        else:
            nnout = self.nnmodel(x)                     # (*BD, nr, 1)

        if isinstance(nnout, tuple):
            nnout, self.aux_loss = nnout
        res = get_out_from_nnout(nnout, n, self.outmultmode,
                                 device=self.device)    # (*BD, nr, 1)

        res = res.squeeze(-1)
        return res

class PureXC(BaseNNXC):
    # BaseNNXC wrapper for xc from libxc
    def __init__(self, xcstr: str):
        super().__init__()
        self.xc = get_xc(xcstr)

    @property
    def family(self) -> int:
        return self.xc.family

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        xc_ene = self.xc.get_edensityxc(densinfo)
        return xc_ene


class PureXCNN(BaseNNXC):
    # neural network xc functional
    def __init__(self, nnmodel: torch.nn.Module, ninpmode: int = 1, outmultmode: int = 1,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.outmultmode = outmultmode
        self.device = device

        self.task = None
        self.aux_loss = None

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # collect the total density (n) and the spin density (xi)
        if isinstance(densinfo, ValGrad):               # unpolarized case
            n = densinfo.value.unsqueeze(-1)            # (*BD, nr, 1)
        else:                                           # polarized case
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd                                 # (*BD, nr, 1)

        # decide how to transform the density to be the input of nn
        ninp = get_n_input(n, self.ninpmode)

        # get the neural network output
        x = ninp.to(self.device)                        # (*BD, nr, 1)
        if self.task:
            nnout = self.nnmodel(x, self.task)          # (*BD, nr, nhid)
        else:
            nnout = self.nnmodel(x)                     # (*BD, nr, 1)

        if isinstance(nnout, tuple):
            nnout, self.aux_loss = nnout
        res = get_out_from_nnout(nnout, n, self.outmultmode,
                                 device=self.device)    # (*BD, nr, 1)
        res = res.squeeze(-1)
        return res

class PicXCNN(BaseNNXC):
    # neural network xc functional
    def __init__(self, nnmodel: torch.nn.Module, ninpmode: int = 1, outmultmode: int = 1,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.outmultmode = outmultmode
        self.device = device

        self.task = None
        self.aux_loss = None

        self.shell_number = 75
        self.angular_points = 302
        dim = 100

        self.shell_weight = nn.Embedding(self.shell_number, dim)  # 设置字典中每个元素的属性向量，向量的维度为dim
        nn.init.ones_(self.shell_weight.weight)  # Initialize each prototype with one.

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # collect the total density (n) and the spin density (xi)
        if isinstance(densinfo, ValGrad):               # unpolarized case
            n = densinfo.value.unsqueeze(-1)            # (*BD, nr, 1)
        else:                                           # polarized case
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd                                 # (*BD, nr, 1)

        # decide how to transform the density to be the input of nn
        ninp = get_n_input(n, self.ninpmode)
        ninps = torch.split(ninp, self.angular_points, dim=0)

        # get the neural network output
        x = ninp.to(self.device)                        # (*BD, nr, 1)
        if self.task:
            nnout = self.nnmodel(x, self.task)          # (*BD, nr, nhid)
        else:
            nnout = self.nnmodel(x)                     # (*BD, nr, 1)

        if isinstance(nnout, tuple):
            nnout, self.aux_loss = nnout
        res = get_out_from_nnout(nnout, n, self.outmultmode,
                                 device=self.device)    # (*BD, nr, 1)
        res = res.squeeze(-1)
        return res


class HybridXC(BaseNNXC):
    def __init__(self, xcstr: str, nnmodel: torch.nn.Module, *,
                 ninpmode: int = 1,         # mode to decide how to transform the density to nn input
                 sinpmode: int = 1,         # mode of calculation of s (normalized gradient) to nn
                                            # (only for GGA or higher)
                 outmultmode: int = 1,      # mode of calculating Eks from output of nn
                 purennmode: bool = False,  # mode of using pure xcnn to caculate edensityxc
                 aweight0: float = 0.0,     # weight of the neural network
                 bweight0: float = 1.0,     # weight of the default xc
                 dtype: torch.dtype = torch.double,
                 device: torch.device = torch.device("cpu"),):

        # hybrid libxc and neural network xc where it starts as libxc and then
        # trains the weights of libxc and nn xc
        super().__init__()
        self.device = device
        self.xc = get_xc(xcstr)             # XC object based on the given expression

        if not purennmode:
            if self.xc.family == 1:
                self.nnxc = NNLDA(nnmodel, ninpmode=ninpmode, outmultmode=outmultmode, device=device)
            elif self.xc.family == 2:
                self.nnxc = NNGGA(nnmodel, ninpmode=ninpmode, sinpmode=sinpmode, outmultmode=outmultmode,
                                  device=device)
        else:
            self.nnxc = PicXCNN(nnmodel, ninpmode=ninpmode, outmultmode=outmultmode, device=device)

        self.aweight = torch.nn.Parameter(torch.tensor(aweight0, dtype=dtype, device=device, requires_grad=True))
        self.bweight = torch.nn.Parameter(torch.tensor(bweight0, dtype=dtype, device=device, requires_grad=True))
        self.weight_activation = torch.nn.Identity()

    # upgrad task tower model for different task
    def upgrad_nnxc(self, task: str):
        self.nnxc.task = task

    @property
    def family(self) -> int:
        return self.xc.family

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # Returns the xc energy density (energy per unit volume)
        nnlda_ene = self.nnxc.get_edensityxc(densinfo)
        lda_ene = self.xc.get_edensityxc(densinfo).to(self.device)
        aweight = self.weight_activation(self.aweight)
        bweight = self.weight_activation(self.bweight)
        return nnlda_ene * aweight + lda_ene * bweight


##################### supporting functions #####################
def get_n_input(n: torch.Tensor, ninpmode: int) -> torch.Tensor:
    # transform the density to the input of the neural network
    if ninpmode == 1:
        return n
    elif ninpmode == 2:
        return safepow(n, 1.0 / 3)
    elif ninpmode == 3:
        return torch.log1p(n)
    else:
        raise RuntimeError("Unknown ninpmode: %d" % ninpmode)

def get_out_from_nnout(nnout: torch.Tensor, n: torch.Tensor, outmultmode: int,
                       device: torch.device = torch.device("cpu")) -> torch.Tensor:
    # calculate the energy density per volume given the density and output
    # of the neural network
    n = n.to(device)
    if outmultmode == 1:
        return nnout * n
    elif outmultmode == 2:
        b = -0.7385587663820223  # -0.75 / np.pi * (3*np.pi**2)**(1./3) for exunif
        exunif = b * safepow(n, 1.0 / 3)
        return nnout * n * exunif
    else:
        raise RuntimeError("Unknown outmultmode: %d" % outmultmode)
