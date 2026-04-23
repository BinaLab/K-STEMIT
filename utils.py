"""Utility helpers for dataset IO, split generation, metrics, and checkpoint saving."""

import dill
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import torch
from tqdm import tqdm
import copy
import os


DEFAULT_SPLIT_COUNT = 5
DEFAULT_SPLIT_SEED = 1337


def generate_splits(dataset_length, split_file, seed=DEFAULT_SPLIT_SEED, split_count=DEFAULT_SPLIT_COUNT):
    if dataset_length <= 0:
        raise ValueError("Dataset must contain at least one sample to create splits.")

    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_length, dtype=np.int64)
    splits = np.zeros((split_count, dataset_length), dtype=np.int64)

    for i in range(split_count):
        splits[i] = rng.permutation(indices)

    np.save(split_file, splits)
    return splits


def ensure_splits(dataset_length, split_file, seed=DEFAULT_SPLIT_SEED, split_count=DEFAULT_SPLIT_COUNT, overwrite=False):
    expected_shape = (split_count, dataset_length)

    if not overwrite and os.path.isfile(split_file):
        splits = np.load(split_file)
        if splits.shape == expected_shape:
            return split_file, False

    split_dir = os.path.dirname(split_file)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)

    generate_splits(dataset_length, split_file, seed=seed, split_count=split_count)
    return split_file, True

def split_dataset(DATASET, SPLIT, SPLIT_FILE):
    splits = np.load(SPLIT_FILE)[SPLIT]

    shuffled_dataset = []
    for index in splits:
        shuffled_dataset.append(copy.deepcopy(DATASET[int(index)]))

    va_index = int(len(shuffled_dataset) * 0.6)
    te_index = int(len(shuffled_dataset) * 0.8)
    return shuffled_dataset[:va_index], shuffled_dataset[va_index:te_index], shuffled_dataset[te_index:]


def load_dill(path):
    with open(path, "rb") as dill_file:
        return dill.load(dill_file)

def rmse(preds, reals):
    return np.sqrt(np.mean((preds-reals)**2))
def rmse_dim(preds, reals, dim):
    return np.sqrt(np.mean((preds-reals)**2, axis=dim))


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.5, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]
    
def save_model(model, args, SPLIT, accelerator):
    torch.save(accelerator.unwrap_model(model).state_dict(), args.EXPERIMENT_NAME + "/model_" + str(SPLIT) + ".pt")
