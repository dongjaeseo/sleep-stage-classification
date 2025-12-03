import numpy as np
import torch
from torch.utils.data import Dataset
import argparse

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy.typing as npt


import re, glob, pickle, argparse, random, numpy as np, torch
from momentfm import MOMENTPipeline
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import os, re, glob, pickle, argparse, random
import pprint
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import sys, inspect
import matplotlib.pyplot as plt
import numpy.typing as npt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score, GridSearchCV

from configs import EXPERIMENTS

'''
--------------------------------------------------------------------------------
What this script does
--------------------------------------------------------------------------------

1. Loads the desired configuration from configs.py through --exp <name>.
2. Loads PSG data from memmap files and converts each sample into 7 chunks
   of length 512 (stride=2, trimming 1s from edges).
3. Builds DataLoaders for training and evaluation.

4. Depending on probe_mode:

   (A) Linear Probe:
       - Load MOMENT in classification mode.
       - Freeze encoder blocks depending on 'freeze'.
       - Train only the classification head.
       - Evaluate using chunk-level logits grouped into 7 chunks per epoch.

   (B) SVM Probe:
       - Load MOMENT in embedding mode.
       - Extract embeddings for all chunks.
       - Train an SVM classifier (direct or with GridSearchCV).
       - Group 7 chunk-level embeddings → one epoch prediction.

5. Reports accuracy, macro-F1, and per-class F1 at epoch level.


--------------------------------------------------------------------------------
Configuration System
--------------------------------------------------------------------------------
Each experiment is defined in CONFIGS.py under EXPERIMENTS, with keys:

    ch_idx        : Which PSG channels to use (1ch / 3ch / 5ch settings).

    probe_mode    : 
        "linear" → use MOMENT in classification mode, train only the head.
        "SVM"    → use MOMENT in embedding mode, extract features, train SVM.

    freeze        : Whether to freeze the embedder + encoder (linear-probe setting).

    use_subset    : If True, use a small subset of the dataset for debugging.

    epoch         : Number of training epochs for linear probes.

    train_subset, test_subset : Number of samples when use_subset=True.

    seed          : Random seed for reproducibility.
'''

def main(cfg_name: str):
    cfg = EXPERIMENTS[cfg_name]

    # unpack config ONCE
    ch_idx       = cfg["ch_idx"]
    probe_mode   = cfg["probe_mode"]
    freeze       = cfg["freeze"]
    use_subset   = cfg["use_subset"]
    num_epochs   = cfg["epoch"]
    train_subset = cfg["train_subset"]
    test_subset  = cfg["test_subset"]
    seed         = cfg["seed"]

    # seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    if not freeze:
        tr_batch_size = 32
        ts_batch_size = 35
    else:
        tr_batch_size = 512
        ts_batch_size = 490

    epoch = num_epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_n = 710144 # Total number of train epochs of patients
    test_n = 179367 # Total number of test epochs of patients

    '''
    Read from memmap file
    Preprocessing PhysioNet
    1. Stride 2 (7680 -> 3840)
    2. Strip 1 second from each end (30s -> 28s => 3840 -> 3584)
    3. Split into 7 parts (3584 -> 512)

    Returned outputs per item:
    timeseries      : (C_selected, 512)
    input_mask      : (512,)
    label           : int
    '''

    class ClassificationDatasetMemmap(Dataset):
        def __init__(self, x_path, y_path, n_samples, C=13, L=7680, ch_idx=[2, 6, 7], stride=2, mean=None, std=None):
            """
            x_path: memmap numpy file (e.g. 'train_X.npy')
            y_path: memmap numpy file (e.g. 'train_y.npy')
            """
            self.X = np.memmap(x_path, dtype="float32", mode="r", shape=(n_samples, C, L))
            self.y = np.memmap(y_path, dtype="int64",   mode="r", shape=(n_samples,))
            self.n, self.C, self.L = self.X.shape

            self.y = self._transform_labels(self.y)

            self.ch_idx = ch_idx if isinstance(ch_idx, (list, tuple)) else [ch_idx]
            self.stride = stride
            self.mean = mean
            self.std = std
            self.seq_len = 512

            # chunks: (sample_idx, start)
            self.chunks = []
            T_eff = self.L // self.stride
            for idx in range(self.n):
                for i in range(7):
                    start = i * self.seq_len + 128
                    if start + self.seq_len <= T_eff:
                        self.chunks.append((idx, start))

        def _transform_labels(self, t_labels: np.ndarray):
            labels = np.unique(t_labels)
            transform = {l: i for i, l in enumerate(labels)}
            return np.vectorize(transform.get)(t_labels)
        
        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, index):
            item_idx, start = self.chunks[index]

            data = self.X[item_idx]  # (C, L)
            label = self.y[item_idx]

            # 채널 고르기 + stride
            data_sub = data[self.ch_idx, ::self.stride]
            chunk = data_sub[:, start:start + self.seq_len]

            if self.mean is not None and self.std is not None:
                chunk = (chunk - self.mean) / (self.std + 1e-8)

            timeseries = chunk.astype(np.float32)
            timeseries_len = timeseries.shape[1]

            # mask
            input_mask = np.ones(self.seq_len, dtype=np.float32)
            input_mask[: self.seq_len - timeseries_len] = 0

            # padding (앞쪽에 0 채움)
            pad_len = self.seq_len - timeseries_len
            if pad_len > 0:
                timeseries = np.pad(timeseries, ((0, 0), (pad_len, 0)))

            return timeseries, input_mask, int(label)

    # Normalization
    train_mean, train_std = 0.0018993263559744006, 50.95342706974399
    print(f'Mean: {train_mean:.2f}, Std: {train_std:.2f}')

    from torch.utils.data import Subset

    train_x_path = '/ssd/kdpark/dongjae/moment/moment_sleep/train_X.npy'
    train_y_path = '/ssd/kdpark/dongjae/moment/moment_sleep/train_y.npy'
    test_x_path = '/ssd/kdpark/dongjae/moment/moment_sleep/test_X.npy'
    test_y_path = '/ssd/kdpark/dongjae/moment/moment_sleep/test_y.npy'
    train_dataset = ClassificationDatasetMemmap(x_path=train_x_path, y_path=train_y_path, ch_idx=ch_idx, n_samples=train_n, mean = train_mean, std = train_std)
    test_dataset = ClassificationDatasetMemmap(x_path=test_x_path, y_path=test_y_path, ch_idx=ch_idx, n_samples=test_n, mean = train_mean, std = train_std)


    # For testing with small dataset
    # Only used if config.use_subset = True
    def sample_subset(dataset, N, seed=42):
        rng = np.random.default_rng(seed)
        N_eff = min(N, len(dataset))
        idx = rng.choice(len(dataset), size=N_eff, replace=False)
        return Subset(dataset, idx)

    def sample_first(dataset, N):
        N_eff = min(N, len(dataset))
        idx = list(range(N_eff))   # [0, 1, 2, ..., N_eff-1]
        return Subset(dataset, idx)

    N_train = train_subset   # whatever you want for train
    N_test  = test_subset    # whatever you want for test

    if use_subset:
        train_dataset = sample_subset(train_dataset, N_train, seed=42)
        test_dataset  = sample_first(test_dataset,  N_test)  # different seed so you don’t mirror train


    train_loader = DataLoader(train_dataset, batch_size=tr_batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=ts_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # Training epoch function
    # Basically equal to one training epoch
    def train_epoch(model, device, train_dataloader, criterion, optimizer, scheduler, reduction='mean'):
        model.to(device)
        model.train()
        losses = []

        for batch_x, batch_mask, batch_labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            batch_x = batch_x.to(device).float()
            batch_labels = batch_labels.to(device)
            batch_mask = batch_mask.to(device).float()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = model(x_enc=batch_x, input_mask=batch_mask, reduction=reduction)
                loss = criterion(output.logits, batch_labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        return avg_loss

    '''
    Evaluation flow:
    1. Compute chunk-level logits
    2. Reshape chunks into groups (group_size = 7) → one sleep epoch per group
    3. Compute probabilities, then average probabilities across the 7 chunks
    4. Pick the final class based on the averaged probabilities (group-level prediction)

    Batch of chunks → model → logits → group into 7 → average → final predictions
    '''
    def evaluate_epoch(dataloader, model, criterion, device, reduction='mean', group_size=7):
        model.eval()
        model.to(device)

        total_loss, total = 0.0, 0
        all_preds, all_labels = [], []
        num_classes = None

        with torch.no_grad():
            for batch_x, batch_mask, batch_labels in dataloader:
                batch_x = batch_x.to(device).float()
                batch_mask = batch_mask.to(device).float()
                batch_labels = batch_labels.to(device).long()

                out = model(x_enc=batch_x, input_mask=batch_mask, reduction=reduction)
                logits = out if torch.is_tensor(out) else out.logits  # (batch, num_classes)

                if num_classes is None:
                    num_classes = logits.shape[1]

                n_groups = logits.shape[0] // group_size
                if n_groups == 0:
                    continue
                logits = logits[: n_groups * group_size]
                batch_labels = batch_labels[: n_groups * group_size]

                # compute chunk-level loss before grouping
                bs = batch_labels.size(0)
                loss = criterion(logits, batch_labels)
                total_loss += loss.item() * bs
                total += bs

                # group labels
                labels_grouped = batch_labels.view(n_groups, group_size)
                consistent = (labels_grouped.max(dim=1).values == labels_grouped.min(dim=1).values)
                if not consistent.all():
                    print("Warning: found groups with inconsistent labels")
                labels_per_group = labels_grouped[:, 0]

                # group predictions
                probs = F.softmax(logits, dim=-1)
                probs_grouped = probs.view(n_groups, group_size, -1)   # (groups, group_size, num_classes)
                probs_mean = probs_grouped.mean(dim=1)                 # (groups, num_classes)
                preds = probs_mean.argmax(dim=-1)                      # (groups,)

                all_preds.append(preds.cpu())
                all_labels.append(labels_per_group.cpu())

        if len(all_preds) == 0:
            return float("nan"), 0.0, 0.0, np.zeros(num_classes)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        avg_loss = total_loss / max(total, 1)
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None,
                                labels=list(range(num_classes)), zero_division=0)

        return avg_loss, accuracy, macro_f1, per_class_f1


    # For SVM, compute embedding
    def get_embedding(model, dataloader):
        embeddings, labels = [], []
        with torch.no_grad():
            for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
                batch_x = batch_x.to("cuda").float()
                batch_masks = batch_masks.to("cuda")

                output = model(x_enc=batch_x, input_mask=batch_masks) # [batch_size x d_model (=1024)]
                embedding = output.embeddings
                embeddings.append(embedding.detach().cpu().numpy())
                labels.append(batch_labels)        

        embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
        return embeddings, labels


    '''
    Use sklearn.SVM -> SVC
    Classifying based on extracted feature vectors

    Output:
    Trained SVC or best estimator from GridSearchCV
    '''
    def fit_svm(features: npt.NDArray, y: npt.NDArray, MAX_SAMPLES: int = 10000):
        nb_classes = np.unique(y, return_counts=True)[1].shape[0]
        train_size = features.shape[0]

        svm = SVC(C=100000, gamma="scale", probability=True)
        if train_size // nb_classes < 5 or train_size < 50:
            return svm.fit(features, y)
        else:
            grid_search = GridSearchCV(
                svm,
                {
                    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    "kernel": ["rbf"],
                    "degree": [3],
                    "gamma": ["scale"],
                    "coef0": [0],
                    "shrinking": [True],
                    "probability": [True],
                    "tol": [0.001],
                    "cache_size": [200],
                    "class_weight": [None],
                    "verbose": [False],
                    "max_iter": [10000000],
                    "decision_function_shape": ["ovr"],
                },
                cv=5,
                n_jobs=10,
            )
            # If the training set is too large, subsample MAX_SAMPLES examples
            if train_size > MAX_SAMPLES:
                split = train_test_split(
                    features, y, train_size=MAX_SAMPLES, random_state=0, stratify=y
                )
                features = split[0]
                y = split[2]

            grid_search.fit(features, y)
            return grid_search.best_estimator_


    # Model & Classifier selection based on config above
    if probe_mode == 'linear':
        model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'classification',
            'n_channels': len(ch_idx), # number of input channels
            'num_class': 5,
            'freeze_encoder': freeze, # Freeze the patch embedding layer
            'freeze_embedder': freeze, # Freeze the transformer encoder
            'freeze_head': False, # The linear forecasting head must be trained
            'enable_gradient_checkpointing': False,
            # Choose how embedding is obtained from the model: One of ['mean', 'concat']
            # Multi-channel embeddings are obtained by either averaging or concatenating patch embeddings 
            # along the channel dimension. 'concat' results in embeddings of size (n_channels * d_model), 
            # while 'mean' results in embeddings of size (d_model)
            'reduction': 'mean',
        },
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
        )

        model.init()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=epoch * len(train_loader))

        # Training loop
        for i in range(epoch):
            train_loss = train_epoch(model, device, train_loader, criterion, optimizer, scheduler)

            # test (or validation) evaluation
            test_loss, test_acc, macro_f1, per_class_f1 = evaluate_epoch(
                test_loader, model, criterion, device
            )

            print(f"[Epoch {i+1}/{epoch}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print(f"  Per-class F1: {per_class_f1}")

            # Save last model
        # torch.save(model.state_dict(), "/ssd/kdpark/dongjae/moment/moment_sleep/20_model_checkpoint.pth")


    # Model & Classifier selection based on config above
    elif probe_mode =='SVM':
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode
        )
        model.init()

        model.to("cuda").float()

        train_emb, train_labels = get_embedding(model, train_loader)
        test_emb, test_labels = get_embedding(model, test_loader)


        sys.stdout.flush() 
        clf = fit_svm(train_emb, train_labels)
        print(type(clf))
        print(clf.get_params())
        test_probs = clf.predict_proba(test_emb)
        test_labels = np.array(test_labels)

        # Make sure N is divisible by 7
        n_groups = len(test_labels) // 7
        test_probs = test_probs[: n_groups * 7]
        test_labels = test_labels[: n_groups * 7]

        # Group into (groups, 7, num_classes)
        probs_grouped = test_probs.reshape(n_groups, 7, -1)
        labels_grouped = test_labels.reshape(n_groups, 7)

        # Sanity check: all 7 labels in each group should match
        consistent = (labels_grouped.max(axis=1) == labels_grouped.min(axis=1))
        if not consistent.all():
            print("Warning: some groups have inconsistent labels")

        # One label per group
        labels_per_group = labels_grouped[:, 0]

        # Average probs across the 7 slices
        probs_mean = probs_grouped.mean(axis=1)     # (groups, num_classes)
        preds = probs_mean.argmax(axis=1)           # (groups,)

        # Compute metrics at group level
        acc = accuracy_score(labels_per_group, preds)
        per_class_f1 = f1_score(labels_per_group, preds, average=None)
        macro_f1 = f1_score(labels_per_group, preds, average="macro")

        print(f"Grouped accuracy: {acc:.2f}")
        print("Per-class F1:", per_class_f1)
        print("Macro F1:", macro_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment configuration to run",
    )
    args = parser.parse_args()
    main(args.exp)