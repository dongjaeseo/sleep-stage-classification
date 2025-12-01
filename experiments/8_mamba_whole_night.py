import numpy as np
import torch
from torch.utils.data import Dataset

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import re, glob, pickle, argparse, random, numpy as np, torch
from momentfm import MOMENTPipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
import os, re, glob, pickle, argparse, random
import pprint
from torch.utils.data import Dataset, DataLoader
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

# This file ASSUMES that you have preprocessed data
# train_x.npy train_y.npy test_x.npy test_y.npy 
# in the same directory as this file

'''
This codepage specifically tests performance improvement when mamba model is used to learn sequential pattern along sleep epochs.
Directly comparing with the best performance from previous settings (Multi channel(3), Linear Classifier, Frozen encoder)

Here we extract embedding from each sleep epoch (mean is computed from 7 parts of embeddings to generate single embedding from epoch)
And then we group patient-wise epochs, and pad according to the maximum number of epochs.
Then computing Mamba state space model along forward and backward direction, modifies the embeddings of patient
such that the embeddings are influenced by sequential epoch data.

Using linear classifier then achieved macro F1 score of 0.7566 which is a significant increase compared to 0.599 without Mamba
'''

# ---------------- cfg ----------------
DEFAULTS = dict(
    # 2, [2, 6, 7]
    # Specifically using [C3-M2, E1-M2, Chin1-Chin2] in this codepage.
    ch_idx=[2, 6, 7],
    probe_mode = 'SVM',
    freeze = True,
    # use_subset = True 
    # for example run with small subset to ensure smooth running
    use_subset = False,
    epoch = 10,
    train_subset = 4329*7,
    test_subset = 1834*7,
    seed=42
)

ap = argparse.ArgumentParser()
for k, v in DEFAULTS.items():
    ap.add_argument(f"--{k}", type=type(v), default=v)
args = ap.parse_args()

print("===== Arguments =====")
pprint.pprint(vars(args))
print("=====================")

random.seed(args.seed)
np.random.seed(args.seed)

if not args.freeze:
    tr_batch_size = 32
    ts_batch_size = 35
else:
    tr_batch_size = 512
    ts_batch_size = 490

epoch = args.epoch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Total no. of sleep epoch Train/Test
train_n = 710144
test_n = 179367

'''
Read from memmap file
Preprocessing PhysioNet
1. Stride 2 (7680 -> 3840)
2. Strip 1 second from each end (30s -> 28s => 3840 -> 3584)
3. Split into 7 parts (3584 -> 512)
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

        # chunks: (sample_idx, start) 목록
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
train_dataset = ClassificationDatasetMemmap(x_path=train_x_path, y_path=train_y_path, ch_idx=args.ch_idx, n_samples=train_n, mean = train_mean, std = train_std)
test_dataset = ClassificationDatasetMemmap(x_path=test_x_path, y_path=test_y_path, ch_idx=args.ch_idx, n_samples=test_n, mean = train_mean, std = train_std)

def sample_subset(dataset, N, seed=42):
    rng = np.random.default_rng(seed)
    N_eff = min(N, len(dataset))
    idx = rng.choice(len(dataset), size=N_eff, replace=False)
    return Subset(dataset, idx)

def sample_first(dataset, N):
    N_eff = min(N, len(dataset))
    idx = list(range(N_eff))   # [0, 1, 2, ..., N_eff-1]
    return Subset(dataset, idx)

N_train = args.train_subset   # whatever you want for train
N_test  = args.test_subset    # whatever you want for test

if args.use_subset:
    train_dataset = sample_subset(train_dataset, N_train, seed=42)
    test_dataset  = sample_first(test_dataset,  N_test)

train_loader = DataLoader(train_dataset, batch_size=tr_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=ts_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)


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


model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode
)
model.init()

model.to("cuda").float()

train_emb, train_labels = get_embedding(model, train_loader)
test_emb, test_labels = get_embedding(model, test_loader)

import pickle, os, glob, re
from tqdm import tqdm

EPOCH_RE = re.compile(r"_(\d+)\.npy$")
x_root = "/ssd/kdpark/sleepfm-codebase/physionet_final/X"
y_root = "/ssd/kdpark/sleepfm-codebase/physionet_final/Y"

# reload patient splits
with open("/ssd/kdpark/sleepfm-codebase/physionet_final/dataset.pickle", "rb") as f:
    split_file = pickle.load(f)

train_list, test_list = [], []
for tr in split_file["train"] + split_file["valid"]:
    for k in tr.keys():
        train_list.append(k)
for tt in split_file["test"]:
    for k in tt.keys():
        test_list.append(k)
train_set = set(train_list)
test_set  = set(test_list)

# count epochs per patient
train_lengths, test_lengths = [], []
pdirs = sorted(d for d in glob.glob(os.path.join(x_root, "*")) if os.path.isdir(d))

for pdir in tqdm(pdirs):
    pid = os.path.basename(pdir)
    x_files = sorted(glob.glob(os.path.join(pdir, f"{pid}_*.npy")))
    if not x_files:
        continue
    if pid in train_set:
        train_lengths.append(len(x_files))
    elif pid in test_set:
        test_lengths.append(len(x_files))

k = 7

# train_lengths = train_lengths[:5]
# test_lengths = test_lengths[:2]
# AFTER
total_epochs_tr = sum(train_lengths)
total_epochs_te = sum(test_lengths)

print(total_epochs_tr)
print(total_epochs_te)

# if labels are per-epoch (recommended)
# Sanity check
assert len(train_emb) == total_epochs_tr * k,  "train embeddings length != epochs*k"
assert len(test_emb)  == total_epochs_te * k,  "test embeddings length != epochs*k"
assert len(train_labels) == total_epochs_tr * k
assert len(test_labels)  == total_epochs_te * k

import numpy as np
from scipy import stats

'''
Each epoch is segmented into 7 parts.
Using fold_k_by_counts, 
7 embeddings constituting one sleep epoch are computed to calculate mean
Therefore providing single embedding from one epoch
'''
def fold_k_by_counts(emb, labels, counts, k=7, reduce='mean', labels_are_per_epoch=False, label_mode='mode', strict=True):
    """
    emb:    [N*k, D] flattened sub-epoch embeddings
    labels: [N] if labels_are_per_epoch else [N*k]
    counts: per-night epoch counts (sum == N)
    k:      sub-epochs per epoch (7)
    reduce: 'mean' or 'median' to collapse 7→1
    label_mode: 'first' or 'mode' if labels are per sub-epoch
    strict: if True, error if a night isn't exactly c*k long; else truncate to multiples of k
    Returns: Xn(list of [T_i,D]), yn(list of [T_i]), new_counts(list[int])
    """
    emb = np.asarray(emb); D = emb.shape[1]
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
            if c_fit == 0: break
            need = c_fit * k
            c = c_fit

        seg = emb[e_ptr:e_ptr+need]               # [c*k, D]
        folded = seg.reshape(c, k, D).mean(axis=1) if reduce == 'mean' else np.median(seg.reshape(c,k,D), axis=1)

        if labels_are_per_epoch:
            y_seg = labels[l_ptr:l_ptr+c]         # [c]
            l_ptr += c
        else:
            lab = labels[l_ptr:l_ptr+need].reshape(c, k)
            if label_mode == 'mode':
                y_seg = stats.mode(lab, axis=1, keepdims=False)[0]
            elif label_mode == 'first':
                y_seg = lab[:, 0]
            else:
                raise ValueError("label_mode must be 'first' or 'mode'")
            l_ptr += need

        Xn.append(folded.astype(np.float32))      # [c, D]
        yn.append(y_seg.astype(np.int64))         # [c]
        new_counts.append(int(c))
        e_ptr += need

    return Xn, yn, new_counts

# Collapse 7 sub-epoch embeddings → 1 epoch embedding per night
k = 7  # 7 sub-epoch embeddings per 30s epoch

# embeddings are length sum(counts)*k, labels are ALSO duplicated to length sum(counts)*k
Xn_train, yn_train, train_counts_folded = fold_k_by_counts(
    train_emb, train_labels, train_lengths,
    k=k, reduce='mean',
    labels_are_per_epoch=False,   
    label_mode='mode',           
    strict=True
)

Xn_test, yn_test, test_counts_folded = fold_k_by_counts(
    test_emb, test_labels, test_lengths,
    k=k, reduce='mean',
    labels_are_per_epoch=False,  
    label_mode='mode',
    strict=True
)


# 1) ---- Build night-level datasets ----
'''
Sequence Learning
Nights class basically groups all epochs of one single patient
returns
[Epoch embedding 1, Epoch embedding 2, ...] for one whole night
[label 1, label 2, ...] the same way
'''
class Nights(Dataset): 
    def __init__(self, X_list, y_list): 
        self.X, self.y = X_list, y_list 
    def __len__(self): 
        return len(self.X) 
    def __getitem__(self, i): 
        return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.long) 
    
'''
From variable length nights for patients,
turns them to padded tensors so the model can process
returns padded X, Y, and mask to represent padded region. 
'''
def collate(batch): 
    xs, ys = zip(*batch) 
    T = max(x.shape[0] for x in xs); D = xs[0].shape[1]; B = len(xs) 
    Xp = torch.zeros(B, T, D, dtype=torch.float32) 
    Yp = torch.full((B, T), -100, dtype=torch.long) 
    M = torch.zeros(B, T, dtype=torch.bool) 
    for i,(x,y) in enumerate(batch): 
        t = x.shape[0]; Xp[i,:t]=x; Yp[i,:t]=y; 
        M[i,:t]=True 
    return Xp, Yp, M 

'''
Mamba model for sequence learning
'''
class MambaStack(nn.Module): 
    def __init__(self, d_model, depth=2, p_drop=0.1): 
        super().__init__() 
        from mamba_ssm import Mamba 
        self.layers = nn.ModuleList([Mamba(d_model) for _ in range(depth)]) 
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)]) 
        self.drop = nn.Dropout(p_drop) 
        
    def forward(self, x, mask=None): 
        if mask is not None: x = x.masked_fill(~mask.unsqueeze(-1), 0.0) 
        for ln, layer in zip(self.norms, self.layers): 
            h = layer(ln(x)) 
            h = self.drop(h) 
            x = x + h 
            if mask is not None: x = x.masked_fill(~mask.unsqueeze(-1), 0.0) 
        return x 
    
'''
Bidirectional Mamba-based head: per-timestep sequence encoder + classifier
'''
class SeqHead(nn.Module): 
    def __init__(self, in_dim=1024, n_classes=5, d_model=256, depth=2, bidi=True, p_drop=0.1): 
        super().__init__() 
        self.bidi = bidi 
        self.inp = nn.Linear(in_dim, d_model) 
        self.fwd = MambaStack(d_model, depth, p_drop) 
        self.bwd = MambaStack(d_model, depth, p_drop) if bidi else None 
        out_dim = d_model * (2 if bidi else 1) 
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, n_classes)) 
        
    def forward(self, x, mask): 
        x = self.inp(x) 
        xf = self.fwd(x, mask) 
        if self.bidi: 
            xr = torch.flip(x, [1]); mr = torch.flip(mask, [1]) 
            xb = self.bwd(xr, mr) 
            xb = torch.flip(xb, [1]) 
            x = torch.cat([xf, xb], dim=-1) 
        else: 
            x = xf 
        logits = self.head(x) 
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e30) 
        return logits 

def masked_ce(logits, targets, mask, class_weights=None, label_smooth=0.05): 
    B,T,C = logits.shape 
    logits, targets, mask = logits.view(B*T, C), targets.view(B*T), mask.view(B*T) 
    if label_smooth and label_smooth > 0: 
        with torch.no_grad(): 
            true = torch.zeros_like(logits); true.fill_(label_smooth/(C-1)) 
            true.scatter_(1, torch.clamp(targets, min=0).unsqueeze(1), 1-label_smooth) 
        loss = -(true * F.log_softmax(logits, dim=-1)).sum(dim=1) 
    else: 
        loss = F.cross_entropy(logits, targets, reduction="none", weight=class_weights) 
    return loss[mask].mean() 

n = len(Xn_train); cut = max(1, int(0.9 * n)) 
dl_tr = DataLoader(Nights(Xn_train[:cut], yn_train[:cut]), batch_size=2, shuffle=True, collate_fn=collate) 
dl_val = DataLoader(Nights(Xn_train[cut:], yn_train[cut:]), batch_size=2, shuffle=False, collate_fn=collate) 
dl_te = DataLoader(Nights(Xn_test, yn_test), batch_size=2, shuffle=False, collate_fn=collate) 

device = "cuda" 
model = SeqHead(in_dim=1024, n_classes=5, d_model=256, depth=2, bidi=True, p_drop=0.1).to(device) 
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01) 

use_metric = "acc" # change to "loss" if you prefer minimizing val_loss 
mode = "max" if use_metric == "acc" else "min" 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( opt, mode=mode, factor=0.3, patience=5, min_lr=3e-5 ) 

best_state, best_metric = None, -float("inf") if use_metric=="acc" else float("inf") 
patience_es = 10 # early-stopping patience 
since_improve = 0 

EPOCHS = 100 
for ep in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    tot_loss, tot_tokens = 0.0, 0
    for xb, yb, mb in dl_tr:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        opt.zero_grad()
        out = model(xb, mb)
        loss = masked_ce(out, yb, mb, label_smooth=0.05)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += loss.item() * mb.sum().item()
        tot_tokens += mb.sum().item()

    # ---- validation ----
    model.eval()
    val_c = val_t = 0
    val_loss_sum = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for xb, yb, mb in dl_val:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            logits = model(xb, mb)
            pred = logits.argmax(-1)
            mask = mb

            # accuracy and loss
            val_c += (pred[mask] == yb[mask]).sum().item()
            val_t += mask.sum().item()
            vloss = masked_ce(logits, yb, mb, label_smooth=0.05)
            val_loss_sum += vloss.item() * mb.sum().item()

            # collect for F1
            all_preds.extend(pred[mask].cpu().numpy())
            all_trues.extend(yb[mask].cpu().numpy())

    train_loss = tot_loss / max(1, tot_tokens)
    val_acc = val_c / max(1, val_t)
    val_loss = val_loss_sum / max(1, val_t)
    val_macro_f1 = f1_score(all_trues, all_preds, average='macro')
    cur_lr = opt.param_groups[0]['lr']

    metric = val_acc if use_metric == "acc" else (-val_loss)
    scheduler.step(val_acc if use_metric == "acc" else val_loss)

    improved = metric > best_metric
    if improved:
        best_metric = metric
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        since_improve = 0
    else:
        since_improve += 1

    print(
        f"ep {ep:03d} | lr {cur_lr:.2e} | "
        f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
        f"val_acc {val_acc:.4f} | val_f1 {val_macro_f1:.4f} | "
        f"best_{use_metric} {best_metric:.4f} | no_improve {since_improve}"
    )

    if since_improve >= patience_es:
        print(f"Early stop at epoch {ep} (no improvement for {patience_es} epochs).")
        break

# ---- restore best ----
if best_state is not None:
    model.load_state_dict(best_state)

# ---- test ----
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for xb, yb, mb in dl_te:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        pred = model(xb, mb).argmax(-1)
        mask = mb
        all_preds.extend(pred[mask].cpu().numpy())
        all_trues.extend(yb[mask].cpu().numpy())

test_acc = (np.array(all_preds) == np.array(all_trues)).mean()
test_macro_f1 = f1_score(all_trues, all_preds, average='macro')
print(f"Test acc: {test_acc:.4f} | Test macro-F1: {test_macro_f1:.4f}")

# ep 001 | lr 1.00e-03 | train_loss 1.2139 | val_loss 1.1978 | val_acc 0.5795 | val_f1 0.4970 | best_acc 0.5795 | no_improve 0
# ep 002 | lr 1.00e-03 | train_loss 0.9752 | val_loss 0.9692 | val_acc 0.6594 | val_f1 0.5948 | best_acc 0.6594 | no_improve 0
# ep 003 | lr 1.00e-03 | train_loss 0.8882 | val_loss 0.8760 | val_acc 0.6977 | val_f1 0.6691 | best_acc 0.6977 | no_improve 0
# ep 004 | lr 1.00e-03 | train_loss 0.8417 | val_loss 0.8544 | val_acc 0.7320 | val_f1 0.6895 | best_acc 0.7320 | no_improve 0
# ep 005 | lr 1.00e-03 | train_loss 0.8230 | val_loss 0.8807 | val_acc 0.7050 | val_f1 0.6921 | best_acc 0.7320 | no_improve 1
# ep 006 | lr 1.00e-03 | train_loss 0.8064 | val_loss 0.8185 | val_acc 0.7374 | val_f1 0.7227 | best_acc 0.7374 | no_improve 0
# ep 007 | lr 1.00e-03 | train_loss 0.7850 | val_loss 0.8170 | val_acc 0.7374 | val_f1 0.6956 | best_acc 0.7374 | no_improve 1
# ep 008 | lr 1.00e-03 | train_loss 0.7741 | val_loss 0.7616 | val_acc 0.7615 | val_f1 0.7405 | best_acc 0.7615 | no_improve 0
# ep 009 | lr 1.00e-03 | train_loss 0.7624 | val_loss 0.8132 | val_acc 0.7372 | val_f1 0.7211 | best_acc 0.7615 | no_improve 1
# ep 010 | lr 1.00e-03 | train_loss 0.7517 | val_loss 0.7532 | val_acc 0.7635 | val_f1 0.7414 | best_acc 0.7635 | no_improve 0
# ep 011 | lr 1.00e-03 | train_loss 0.7488 | val_loss 0.7670 | val_acc 0.7671 | val_f1 0.7275 | best_acc 0.7671 | no_improve 0
# ep 012 | lr 1.00e-03 | train_loss 0.7387 | val_loss 0.7534 | val_acc 0.7614 | val_f1 0.7333 | best_acc 0.7671 | no_improve 1
# ep 013 | lr 1.00e-03 | train_loss 0.7296 | val_loss 0.7697 | val_acc 0.7528 | val_f1 0.7381 | best_acc 0.7671 | no_improve 2
# ep 014 | lr 1.00e-03 | train_loss 0.7210 | val_loss 0.7675 | val_acc 0.7648 | val_f1 0.7342 | best_acc 0.7671 | no_improve 3
# ep 015 | lr 1.00e-03 | train_loss 0.7092 | val_loss 0.7921 | val_acc 0.7553 | val_f1 0.7306 | best_acc 0.7671 | no_improve 4
# ep 016 | lr 1.00e-03 | train_loss 0.7113 | val_loss 0.7545 | val_acc 0.7657 | val_f1 0.7446 | best_acc 0.7671 | no_improve 5
# ep 017 | lr 1.00e-03 | train_loss 0.6953 | val_loss 0.7606 | val_acc 0.7611 | val_f1 0.7275 | best_acc 0.7671 | no_improve 6
# ep 018 | lr 3.00e-04 | train_loss 0.6545 | val_loss 0.7378 | val_acc 0.7707 | val_f1 0.7476 | best_acc 0.7707 | no_improve 0
# ep 019 | lr 3.00e-04 | train_loss 0.6404 | val_loss 0.7352 | val_acc 0.7740 | val_f1 0.7457 | best_acc 0.7740 | no_improve 0
# ep 020 | lr 3.00e-04 | train_loss 0.6308 | val_loss 0.7763 | val_acc 0.7529 | val_f1 0.7373 | best_acc 0.7740 | no_improve 1
# ep 021 | lr 3.00e-04 | train_loss 0.6201 | val_loss 0.7657 | val_acc 0.7622 | val_f1 0.7425 | best_acc 0.7740 | no_improve 2
# ep 022 | lr 3.00e-04 | train_loss 0.6115 | val_loss 0.7783 | val_acc 0.7581 | val_f1 0.7306 | best_acc 0.7740 | no_improve 3
# ep 023 | lr 3.00e-04 | train_loss 0.6033 | val_loss 0.7788 | val_acc 0.7641 | val_f1 0.7420 | best_acc 0.7740 | no_improve 4
# ep 024 | lr 3.00e-04 | train_loss 0.5931 | val_loss 0.7776 | val_acc 0.7615 | val_f1 0.7355 | best_acc 0.7740 | no_improve 5
# ep 025 | lr 3.00e-04 | train_loss 0.5857 | val_loss 0.8002 | val_acc 0.7593 | val_f1 0.7359 | best_acc 0.7740 | no_improve 6
# ep 026 | lr 9.00e-05 | train_loss 0.5671 | val_loss 0.7981 | val_acc 0.7607 | val_f1 0.7381 | best_acc 0.7740 | no_improve 7
# ep 027 | lr 9.00e-05 | train_loss 0.5604 | val_loss 0.8063 | val_acc 0.7578 | val_f1 0.7337 | best_acc 0.7740 | no_improve 8
# ep 028 | lr 9.00e-05 | train_loss 0.5562 | val_loss 0.8134 | val_acc 0.7570 | val_f1 0.7331 | best_acc 0.7740 | no_improve 9
# ep 029 | lr 9.00e-05 | train_loss 0.5526 | val_loss 0.8167 | val_acc 0.7575 | val_f1 0.7317 | best_acc 0.7740 | no_improve 10
# Early stop at epoch 29 (no improvement for 10 epochs).
# Test acc: 0.7876 | Test macro-F1: 0.7566
