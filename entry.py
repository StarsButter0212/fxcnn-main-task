from __future__ import annotations
import os
from abc import abstractmethod, abstractproperty
import functools
from typing import List, Dict, Callable, Optional, Union
import hashlib
import pickle
import numpy as np
import torch
from pyscf import gto, cc, scf
from pyscf.dft import numint
import dqc
from dqc.utils.datastruct import SpinParam
from dqc.system.mol import Mol
from dqc.system.base_system import BaseSystem
from dqc.grid.base_grid import BaseGrid
from kscalc import BaseKSCalc
from xcmodels import BaseNNXC
from utils import eval_and_save

filedir = os.path.dirname(os.path.realpath(__file__))


class System(dict):
    """
    Interface to the system in the dataset.
    No scientific calculation should be performed in this class.
    Please do not initialize this class directly, instead, use
    ``System.create()``.
    """

    created_systems: Dict[str, System] = {}

    @classmethod
    def create(cls, system: Dict) -> System:
        # create the system if it has not been created
        # otherwise, return the previously created system

        system_str = str(system)
        if system_str not in cls.created_systems:
            cls.created_systems[system_str] = System(system)
        return cls.created_systems[system_str]

    def __init__(self, system: Dict):
        super().__init__(system)

        # caches
        self._caches = {}

    def get_pyscf_system(self):
        # convert the system dictionary PySCF system

        systype = self["type"]
        if systype == "mol":
            kwargs = self["kwargs"]
            return gto.M(atom=kwargs["moldesc"], basis=kwargs["basis"],
                         spin=kwargs.get("spin", 0), unit="Bohr",
                         charge=kwargs.get("charge", 0))
        else:
            raise RuntimeError("Unknown system type: %s" % systype)

    def get_dqc_system(self, pos_reqgrad: bool = False, grid: Union[int, str] = 'sg3',
                       truncate: Optional[str] = 'dasgupta',
                       device: torch.device = torch.device('cpu')) -> BaseSystem:

        # convert the system dictionary to DQC system
        systype = self["type"]
        if systype == "mol":
            atomzs, atomposs = dqc.parse_moldesc(self["kwargs"]["moldesc"],
                                                 device=device)
            if pos_reqgrad:
                atomposs.requires_grad_()
            mol = Mol(**self["kwargs"], grid=grid, truncate=truncate, device=device)
            return mol
        else:
            raise RuntimeError("Unknown system type: %s" % systype)

    def get_cache(self, s: str):
        return self._caches.get(s, None)

    def set_cache(self, s: str, obj) -> None:
        self._caches[s] = obj

######################## entry ########################

class Entry(dict):
    """
    Interface to the entry of the dataset.
    Entry class should not be initialized directly, but created through
    ``Entry.create``
    """
    created_entries: Dict[str, Entry] = {}

    @classmethod
    def create(cls, entry_dct: Union[Dict, Entry],
               dtype: torch.dtype = torch.double,
               device: torch.device = torch.device('cpu')) -> Entry:

        cls.device = device
        if isinstance(entry_dct, Entry):
            # TODO: should we add dtype and device checks here?
            return entry_dct

        s = str(entry_dct)
        if s not in cls.created_entries:
            tpe = entry_dct["type"]
            kwargs = {
                "entry_dct": entry_dct,
                "dtype": dtype,
                "device": device,
            }
            obj = {
                "ie": EntryIE,
                "ae": EntryAE,
                "dm": EntryDM,
                "dens": EntryDens,
                "force": EntryForce,
            }[tpe](**kwargs)
            cls.created_entries[s] = obj

        return cls.created_entries[s]

    def __init__(self, entry_dct: Dict,
                 dtype: torch.dtype = torch.double,
                 device: torch.device = torch.device('cpu')):
        super().__init__(entry_dct)

        self._systems = [System.create(p) for p in entry_dct["systems"]]
        self._dtype = dtype
        self._device = device

        self._trueval_is_set = False
        self._trueval = torch.tensor(0.0)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def get_systems(self) -> List[System]:
        """
        Returns the list of systems in the entry
        """
        return self._systems

    @abstractproperty
    def entry_type(self) -> str:
        """
        Returning the type of the entry of the dataset
        """
        pass

    def get_true_val(self, file_dir: str = None) -> torch.Tensor:
        if not self._trueval_is_set:
            if file_dir:
                self._trueval = self._get_true_val(file_dir)
            else:
                self._trueval = self._get_true_val()
            self._trueval_is_set = True
        return self._trueval

    @abstractmethod
    def _get_true_val(self, file_dir: str = None) -> torch.Tensor:
        """
        Get the true value of the entry.
        """
        pass

    @abstractmethod
    def get_val(self, qcs: List[BaseKSCalc]) -> torch.Tensor:
        """
        Calculate the value of the entry given post-run QC objects.
        """
        pass

    @abstractmethod
    def get_loss(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        """
        Returns the unweighted loss function of the entry based on the value
        and true value supplied.
        """
        pass

    @abstractmethod
    def get_deviation(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        """
        Returns the deviation of predicted value and true value in an interpretable
        format and units.
        """
        pass

    @abstractmethod
    def get_true_dm(self, file_dir, system) -> torch.Tensor:
        """
        Get the true value of the entry.
        """
        pass

class EntryIE(Entry):
    """Entry for Ionization Energy (IE)"""
    @property
    def entry_type(self) -> str:
        return "ie"

    def _get_true_val(self, file_dir: str = None) -> torch.Tensor:
        return torch.as_tensor(self["true_val"], dtype=self.dtype, device=self.device)

    def get_val(self, qcs: List[BaseKSCalc]) -> torch.Tensor:
        glob = {
            "systems": qcs,
            "energy": self.energy
        }
        return eval(self["cmd"], glob)

    def get_loss(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        return torch.mean((val - true_val) ** 2)    # MSE

    def get_deviation(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        return (val - true_val).abs() * 627.5       # MAE in kcal/mol  > 1 Hartree (au) = 627.51 kcal/mol

    def energy(self, qc: BaseKSCalc) -> torch.Tensor:
        return qc.energy()

    @eval_and_save
    def get_true_dm(self, file_dir, system) -> torch.Tensor:
        # get the density matrix from PySCF's CCSD calculation
        return self.calc_pyscf_dm_tot(system)

    @classmethod
    def calc_pyscf_dm_tot(cls, system: System):
        # run the PySCF's CCSD calculation (for the true value calculation)
        mol = system.get_pyscf_system()
        mf = scf.UHF(mol).run()
        mcc = cc.UCCSD(mf)
        mcc.kernel()

        # obtain the total density matrix
        modm = mcc.make_rdm1()
        aodm0 = np.dot(mf.mo_coeff[0], np.dot(modm[0], mf.mo_coeff[0].T))
        aodm1 = np.dot(mf.mo_coeff[1], np.dot(modm[1], mf.mo_coeff[1].T))

        return torch.as_tensor([aodm0, aodm1], dtype=torch.double)

class EntryAE(EntryIE):
    """Entry for Atomization Energy (AE)"""
    @property
    def entry_type(self) -> str:
        return "ae"

class EntryDens(Entry):
    """Entry for density profile (dens), compared with CCSD calculation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.get_systems()) == 1, "dens entry can only have 1 system"
        self._grid: Optional[BaseGrid] = None

    @property
    def entry_type(self) -> str:
        return "dens"

    @eval_and_save
    def _get_true_val(self, file_dir: str = None) -> torch.Tensor:
        # get the density profile from PySCF's CCSD calculation

        # get the density matrix from the PySCF calculation
        system = self.get_systems()[0]
        dm = EntryDM.calc_pyscf_dm_tot(system)

        # get the grid from dqc and its weights with the reduced level of the grid
        rgrid = self._get_integration_grid().get_rgrid()    # (*BG, ngrid, ndim)
        rgrid = rgrid.cpu()                                 # customer-added

        # calculate the density profile
        mol = system.get_pyscf_system()
        ao = numint.eval_ao(mol, rgrid)
        dens = numint.eval_rho(mol, ao, dm)                 # (*BG, ngrid)
        return torch.as_tensor(dens, dtype=self.dtype, device=self.device)

    def get_val(self, qcs: List[BaseKSCalc]) -> torch.Tensor:
        qc = qcs[0]
        self._grid = qc.qc._engine._system.get_grid()

        # get the density profile
        return qc.dens()

    def get_loss(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        # integration of squared difference at all spaces
        dvol = self._get_integration_grid().get_dvolume().to(self.device)
        return torch.sum((true_val - val) ** 2 * dvol)

    def get_deviation(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        return self.get_loss(val, true_val) * 1000.0  # sum of squares in 10^-3/Bohr^3

    def _get_integration_grid(self) -> BaseGrid:
        if self._grid is None:
            system = self.get_systems()[0]

            # reduce the level of grid
            orig_grid_level: Optional[int] = None
            if "grid" in system:
                orig_grid_level = system["grid"]

            # get the dqc grid
            system["grid"] = "sg2"
            dqc_mol = system.get_dqc_system()
            dqc_mol.setup_grid()
            grid = dqc_mol.get_grid()
            assert grid.coord_type == "cart"

            # restore the grid level
            if orig_grid_level is not None:
                system["grid"] = orig_grid_level
            else:
                system.pop("grid")

            self._grid = grid

        return self._grid

    @eval_and_save
    def get_true_dm(self, file_dir, system) -> torch.Tensor:
        # get the density matrix from PySCF's CCSD calculation
        return EntryIE.calc_pyscf_dm_tot(system)


class EntryDM(Entry):
    """Entry for Density Matrix (DM)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.get_systems()) == 1, "dm entry can only have 1 system"

    @property
    def entry_type(self) -> str:
        return "dm"

    @eval_and_save
    def _get_true_val(self, file_dir: str = None) -> torch.Tensor:
        # get the density matrix from PySCF's CCSD calculation
        system = self.get_systems()[0]
        return self.calc_pyscf_dm_tot(system)

    def get_val(self, qcs: List[BaseKSCalc]) -> torch.Tensor:
        return qcs[0].aodmtot()

    def get_loss(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        return torch.mean((val - true_val) ** 2)

    def get_deviation(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        return (val - true_val).abs() * 627.5  # MAE in kcal/mol

    @classmethod
    def calc_pyscf_dm_tot(cls, system: System):
        # run the PySCF's CCSD calculation (for the true value calculation)
        mol = system.get_pyscf_system()
        mf  = scf.UHF(mol).run()
        mcc = cc.UCCSD(mf)
        mcc.kernel()

        # obtain the total density matrix
        modm = mcc.make_rdm1()
        aodm0 = np.dot(mf.mo_coeff[0], np.dot(modm[0], mf.mo_coeff[0].T))
        aodm1 = np.dot(mf.mo_coeff[1], np.dot(modm[1], mf.mo_coeff[1].T))
        aodm = aodm0 + aodm1

        return torch.as_tensor(aodm, dtype=torch.double)

class EntryForce(Entry):
    """Entry for force at the experimental equilibrium position"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.get_systems()) == 1, "force entry can only have 1 system"

    @property
    def entry_type(self) -> str:
        return "force"

    def _get_true_val(self, file_dir: str = None) -> torch.Tensor:
        # get the density matrix from PySCF's CCSD calculation
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    def get_val(self, qcs: List[BaseKSCalc]) -> torch.Tensor:
        return qcs[0].force()

    def get_loss(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        return torch.mean((val - true_val) ** 2)

    def get_deviation(self, val: torch.Tensor, true_val: torch.Tensor) -> torch.Tensor:
        return (val - true_val) * 627.5 * 1.88972687777  # MAE in kcal/mol/angstrom
