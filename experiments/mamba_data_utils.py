# experiments/mamba_data_utils.py

"""
Data and sequence utilities for the Mamba sleep experiment.

This file:
- Defines a memmap-based dataset for PSG epochs (ClassificationDatasetMemmap)
- Builds DataLoaders for train/test using preprocessed PhysioNet sleep data
- Computes per-patient epoch counts from SleepFM-style splits
- Folds 7 sub-epoch embeddings into a single epoch embedding
- Wraps per-night epoch sequences into a Dataset (Nights) with a padding collate function
"""

import os
import glob
import pickle
from typing import List, Tuple

import numpy as np
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class ClassificationDatasetMemmap(Dataset):
    """
    Memmap based PSG epoch dataset.
    - X: (n_samples, C, L)
    - y: (n_samples,)
    - Each sample is split into 7 sub-epochs, and the dataset index is constructed as a list of (sample_idx, start) pairs.
    """
    def __init__(
        self,
        x_path: str,
        y_path: str,
        n_samples: int,
        C: int = 13,
        L: int = 7680,
        ch_idx=(2, 6, 7),
        stride: int = 2,
        mean: float | None = None,
        std: float | None = None,
        seq_len: int = 512,
    ):
        self.X = np.memmap(x_path, dtype="float32", mode="r", shape=(n_samples, C, L))
        self.y = np.memmap(y_path, dtype="int64",   mode="r", shape=(n_samples,))
        self.n, self.C, self.L = self.X.shape

        # relabel to 0..(K-1)
        self.y = self._transform_labels(self.y)

        self.ch_idx = list(ch_idx) if isinstance(ch_idx, (list, tuple)) else [ch_idx]
        self.stride = stride
        self.mean = mean
        self.std = std
        self.seq_len = seq_len

        # chunk index: (sample_idx, start)
        self.chunks = []
        T_eff = self.L // self.stride
        for idx in range(self.n):
            for i in range(7):  # 7 sub-epochs per epoch
                start = i * self.seq_len + 128
                if start + self.seq_len <= T_eff:
                    self.chunks.append((idx, start))

    @staticmethod
    def _transform_labels(t_labels: np.ndarray):
        labels = np.unique(t_labels)
        transform = {l: i for i, l in enumerate(labels)}
        return np.vectorize(transform.get)(t_labels)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        item_idx, start = self.chunks[index]

        data = self.X[item_idx]  # (C, L)
        label = int(self.y[item_idx])

        # channel select + stride
        data_sub = data[self.ch_idx, ::self.stride]
        chunk = data_sub[:, start:start + self.seq_len]

        if self.mean is not None and self.std is not None:
            chunk = (chunk - self.mean) / (self.std + 1e-8)

        timeseries = chunk.astype(np.float32)
        timeseries_len = timeseries.shape[1]

        # mask
        input_mask = np.ones(self.seq_len, dtype=np.float32)
        input_mask[: self.seq_len - timeseries_len] = 0

        # left padding
        pad_len = self.seq_len - timeseries_len
        if pad_len > 0:
            timeseries = np.pad(timeseries, ((0, 0), (pad_len, 0)))

        return timeseries, input_mask, label


def sample_subset(dataset, N: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    N_eff = min(N, len(dataset))
    idx = rng.choice(len(dataset), size=N_eff, replace=False)
    return Subset(dataset, idx)


def sample_first(dataset, N: int):
    N_eff = min(N, len(dataset))
    idx = list(range(N_eff))
    return Subset(dataset, idx)


def build_memmap_dataloaders(
    train_x_path: str,
    train_y_path: str,
    test_x_path: str,
    test_y_path: str,
    train_n: int,
    test_n: int,
    ch_idx=(2, 6, 7),
    mean: float = 0.0018993263559744006,
    std: float = 50.95342706974399,
    freeze: bool = True,
    use_subset: bool = False,
    train_subset: int = 4329 * 7,
    test_subset: int = 1834 * 7,
    num_workers: int = 4,
):
    if not freeze:
        tr_batch_size = 32
        ts_batch_size = 35
    else:
        tr_batch_size = 512
        ts_batch_size = 490

    train_dataset = ClassificationDatasetMemmap(
        x_path=train_x_path,
        y_path=train_y_path,
        n_samples=train_n,
        ch_idx=ch_idx,
        mean=mean,
        std=std,
    )
    test_dataset = ClassificationDatasetMemmap(
        x_path=test_x_path,
        y_path=test_y_path,
        n_samples=test_n,
        ch_idx=ch_idx,
        mean=mean,
        std=std,
    )

    if use_subset:
        train_dataset = sample_subset(train_dataset, train_subset, seed=42)
        test_dataset = sample_first(test_dataset, test_subset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=tr_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=ts_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader


# Figures out no. epochs for each patient
def compute_lengths_per_patient(
    x_root: str,
    split_pickle_path: str,
) -> tuple[list[int], list[int]]:
    """
    x_root: /.../physionet_final/X
    split_pickle_path: .../dataset.pickle (sleepfm split)
    return: train_lengths, test_lengths (list of epoch counts per night)
    """
    with open(split_pickle_path, "rb") as f:
        split_file = pickle.load(f)

    train_list, test_list = [], []
    for tr in split_file["train"] + split_file["valid"]:
        for k in tr.keys():
            train_list.append(k)
    for tt in split_file["test"]:
        for k in tt.keys():
            test_list.append(k)

    train_set = set(train_list)
    test_set = set(test_list)

    train_lengths, test_lengths = [], []
    pdirs = sorted(d for d in glob.glob(os.path.join(x_root, "*")) if os.path.isdir(d))

    for pdir in pdirs:
        pid = os.path.basename(pdir)
        x_files = sorted(glob.glob(os.path.join(pdir, f"{pid}_*.npy")))
        if not x_files:
            continue
        if pid in train_set:
            train_lengths.append(len(x_files))
        elif pid in test_set:
            test_lengths.append(len(x_files))

    return train_lengths, test_lengths


"""
Collapse sub-epoch embeddings into full-epoch embeddings and regroup them by night.

For each night:
- Takes k sub-epoch embeddings per epoch (shape [c*k, D])
- Reshapes into [c, k, D] and reduces (mean/median) → [c, D]
- Computes one label per epoch (mode/first)
- Returns a list of night-level sequences: Xn[i] shape [T_i, D], yn[i] shape [T_i]
"""
def fold_k_by_counts(
    emb: np.ndarray,
    labels: np.ndarray,
    counts: List[int],
    k: int = 7,
    reduce: str = "mean",
    labels_are_per_epoch: bool = False,
    label_mode: str = "mode",
    strict: bool = True,
) -> Tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    
    emb = np.asarray(emb)
    D = emb.shape[1]
    labels = np.asarray(labels)
    counts = [int(c) for c in counts]

    Xn, yn, new_counts = [], [], []
    e_ptr = 0
    l_ptr = 0

    total_epochs = sum(counts)
    if labels_are_per_epoch:
        assert len(labels) == total_epochs, "epoch-level labels expected"
    else:
        assert len(labels) == total_epochs * k, "sub-epoch labels expected"

    for c in counts:
        need = c * k
        have = emb.shape[0] - e_ptr
        if have < need:
            if strict:
                raise ValueError(f"Expected {need} sub-epochs, got {have}. Set strict=False to truncate.")
            c_fit = have // k
            if c_fit == 0:
                break
            need = c_fit * k
            c = c_fit

        seg = emb[e_ptr:e_ptr + need]  # [c*k, D]
        seg_reshaped = seg.reshape(c, k, D)

        if reduce == "mean":
            folded = seg_reshaped.mean(axis=1)
        elif reduce == "median":
            folded = np.median(seg_reshaped, axis=1)
        else:
            raise ValueError("reduce must be 'mean' or 'median'")

        if labels_are_per_epoch:
            y_seg = labels[l_ptr:l_ptr + c]  # [c]
            l_ptr += c
        else:
            lab = labels[l_ptr:l_ptr + need].reshape(c, k)
            if label_mode == "mode":
                y_seg = stats.mode(lab, axis=1, keepdims=False)[0]
            elif label_mode == "first":
                y_seg = lab[:, 0]
            else:
                raise ValueError("label_mode must be 'first' or 'mode'")
            l_ptr += need

        Xn.append(folded.astype(np.float32))   # [c, D]
        yn.append(y_seg.astype(np.int64))      # [c]
        new_counts.append(int(c))
        e_ptr += need

    return Xn, yn, new_counts


"""
Dataset where each item is one full-night sequence for one patient.
X_list[i] = epoch embeddings of shape [T_i, D]
y_list[i] = labels of shape [T_i]
Used to feed night-level sequences into the Mamba model.
"""
class Nights(Dataset):
    """
    Night-level dataset: Whole night epochs of 1 patient
    X_list: list of [T_i, D]
    y_list: list of [T_i]
    """
    def __init__(self, X_list, y_list):
        self.X = X_list
        self.y = y_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.long),
        )


"""
Pad a batch of variable-length nights into fixed-size tensors.
Returns:
  Xp : (B, T_max, D)  padded embeddings
  Yp : (B, T_max)     padded labels (unused positions = -100)
  M  : (B, T_max)     mask indicating valid (non-padded) positions
"""
def collate_nights(batch):
    """
    Variable length nights → padded batch.
    Returns: Xp (B,T,D), Yp (B,T), M (B,T) mask
    """
    xs, ys = zip(*batch)
    T = max(x.shape[0] for x in xs)
    D = xs[0].shape[1]
    B = len(xs)

    Xp = torch.zeros(B, T, D, dtype=torch.float32)
    Yp = torch.full((B, T), -100, dtype=torch.long)
    M = torch.zeros(B, T, dtype=torch.bool)

    for i, (x, y) in enumerate(batch):
        t = x.shape[0]
        Xp[i, :t] = x
        Yp[i, :t] = y
        M[i, :t] = True

    return Xp, Yp, M
