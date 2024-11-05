# %%
import logging
import multiprocessing
import os
from functools import partial
import pickle
import torch


class Pipe:
    def __init__(self, raw, functional, progress_check=1000):
        self.raw = raw
        self.functional = functional
        manager = multiprocessing.Manager()
        self.errors = manager.list()
        self.out = []
        try:
            self.functional_name = self.functional.__name__
        except AttributeError:
            self.functional_name = self.functional.__class__.__name__
        logging.info(f"Pipe instance created with functional: {self.functional_name}")
        self.progress_check = progress_check

    def _progress(self, i):
        if self.progress_check > 0 and (i + 1) % self.progress_check == 0:
            torch.cuda.empty_cache()
            logging.info(f": {self.functional_name}: Processing index {i+1}")

    def ff(self, tup):
        i, datapoint = tup
        self._progress(i)
        try:
            ans = self.functional(datapoint)
            return ans
        except Exception as e:
            self.errors.append(f"Error at index {i}: {e}")
            logging.error(f"Error at index {i}: {e}")
            return None

    def run_multiproc(self, cores=None):
        if cores is None:
            cores = multiprocessing.cpu_count()
        with multiprocessing.Pool() as pool:
            data = pool.map(partial(self.ff), zip(range(len(list(self.raw))), self.raw))
        self.out = data
        return self.out

    def run_each(self):
        for i, x in enumerate(self.raw):
            self.out.append(self.ff((i, x)))
        return self.out

    def pickle_list_to(self, pathname):
        foldername = os.path.dirname(pathname)
        os.makedirs(foldername, exist_ok=True)
        file_path = f"{pathname}.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(self.out, file)

    def get_errors(self):
        return list(self.errors)

    def get_out_once(self, i=0):
        return self.ff(i, list(self.raw)[i])
