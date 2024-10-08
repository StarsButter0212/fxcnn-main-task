import os
import re
import torch
import pickle
import hashlib
import functools
from typing import List, Callable, Optional, Union

filedir = os.path.dirname(os.path.realpath(__file__))

def subs_present(cs: List[str], s: Union[str, List[str]], at_start: bool = False) -> bool:
    # find the characters/substrings in the string or list of string
    # return True if at least one of the substring present in the string s
    for c in cs:
        if not at_start:
            if c in s:
                return True
        else:
            if c == s[:len(c)]:
                return True
    return False

def get_atoms(s: str) -> List[str]:
    # returns the atoms in the given molecule's name (e.g. NH2 will return ["N", "H"])
    pattern = r"([A-Z][a-z]*)"
    return re.findall(pattern, s)

def print_active_tensors(printout: bool = True) -> int:
    # NOTE: This function does not work if imported, so you have to copy and paste
    # this code to your main file in order to make it work
    import gc
    npresents = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if printout:
                    print(type(obj), obj.size(), obj.dtype)
                npresents += 1
        except:
            pass
    return npresents

def eval_and_save(fcn: Callable):
    # save the results of the calculation of fcn to a pickle file

    @functools.wraps(fcn)
    def new_fcn(*args, **kwargs):
        # get the representation of args and kwargs
        s = fcn.__name__
        s += ";" + (";".join([str(args[0])]))
        s += ";" + (";".join(["%s=%s" % (k, v) for (k, v) in kwargs.items()]))

        # get the file name to store the evaluated values
        fname = hashstr(s) + ".pkl"
        if len(args) > 1:
            fdir = os.path.join(filedir, ".datasets", args[1])
        else:
            fdir = os.path.join(filedir, ".datasets")

        if not os.path.exists(fdir):
            os.mkdir(fdir)
        fpath = os.path.join(fdir, fname)

        # if the file exists, then load from the file, otherwise evaluate
        if os.path.exists(fpath):
            with open(fpath, "rb") as fb:
                res = pickle.load(fb)
        else:
            # evaluate the true value
            print("Evaluating the true value of '%s' and save it to %s" % \
                  (s, fpath))
            res = fcn(*args, **kwargs)

            # save the result to a file
            with open(fpath, "wb") as fb:
                pickle.dump(res, fb)
        return res
    return new_fcn

def hashstr(s: str) -> str:
    # encode the string into hashed format
    return str(hashlib.blake2s(str.encode(s)).hexdigest())

def get_exp_version(version: Optional[str]) -> Optional[str]:
    # get the experiment version based on the input from the user's hparams
    if version is not None:
        if version.isdigit():
            return "version_%d" % int(version)
    return version

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        #self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        #self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = 0 if self.count == 0 else self.sum / self.count
        return avg