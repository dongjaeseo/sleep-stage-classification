# experiments/mamba_whole_night.py

"""
Main experiment script for Mamba-based sleep-stage classification.

This file:
- Loads preprocessed PSG data via memmap dataloaders
- Extracts epoch embeddings using the MOMENT foundation model
- Groups epochs into per-night sequences
- Trains and evaluates a bidirectional Mamba sequence model on night-level sequences
"""

"""
This code tests whether a Mamba-based sequence model can improve sleep-stage
classification by modeling temporal structure across sleep epochs.

Baseline for comparison:
    • Best previous setting: 3-channel input, linear classifier,
      frozen encoder (no temporal modeling).

Method:
    • Extract embeddings for each epoch (mean of the 7 chunk embeddings).
    • Group epochs by patient and pad sequences to the maximum length.
    • Apply a Mamba state-space model bidirectionally across the epoch sequence.
      This updates each epoch embedding using information from neighboring epochs.

Result:
    • Linear classifier on Mamba-refined embeddings achieved higher macro-F1
      than the no-temporal baseline.
"""



import os
import random
import argparse
import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from momentfm import MOMENTPipeline

from mamba_data_utils import (
    build_memmap_dataloaders,
    compute_lengths_per_patient,
    fold_k_by_counts,
    Nights,
    collate_nights,
)
from mamba_seq_models import SeqHead, masked_ce


DEFAULTS = dict(
    ch_idx=[2, 6, 7],
    freeze=True,
    use_subset=False,
    epoch=10,  # unused here but kept for compatibility
    train_subset=4329 * 7,
    test_subset=1834 * 7,
    seed=42,
    # data paths
    train_x_path="C:/ssd/kdpark/dongjae/moment/moment_sleep/train_X.npy",
    train_y_path="C:/ssd/kdpark/dongjae/moment/moment_sleep/train_y.npy",
    test_x_path="C:/ssd/kdpark/dongjae/moment/moment_sleep/test_X.npy",
    test_y_path="C:/ssd/kdpark/dongjae/moment/moment_sleep/test_y.npy",
    train_n=710144,
    test_n=179367,
    mean=0.0018993263559744006,
    std=50.95342706974399,
    x_root="C:/ssd/kdpark/sleepfm-codebase/physionet_final/X",
    split_pickle="C:/ssd/kdpark/sleepfm-codebase/physionet_final/dataset.pickle",
    k=7,
    bidi=True,
    d_model=256,
    depth=2,
    lr=1e-3,
    weight_decay=0.01,
    label_smooth=0.05,
    epochs=100,
    batch_size_night=2,
    patience_es=10,
)


def parse_args():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args()
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Extract Embedding from MOMENT
def get_embedding(model, dataloader, device="cuda"):
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to(device).float()
            batch_masks = batch_masks.to(device)

            output = model(x_enc=batch_x, input_mask=batch_masks)
            emb = output.embeddings  # [B, 1024]
            embeddings.append(emb.detach().cpu().numpy())
            labels.append(batch_labels.numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels


def main():
    # GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = parse_args()
    print("===== Arguments =====")
    pprint.pprint(vars(args))
    print("=====================")

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Memmap dataloader (sub-epoch level)
    train_loader, test_loader = build_memmap_dataloaders(
        train_x_path=args.train_x_path,
        train_y_path=args.train_y_path,
        test_x_path=args.test_x_path,
        test_y_path=args.test_y_path,
        train_n=args.train_n,
        test_n=args.test_n,
        ch_idx=args.ch_idx,
        mean=args.mean,
        std=args.std,
        freeze=args.freeze,
        use_subset=args.use_subset,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    print(f"Mean: {args.mean:.2f}, Std: {args.std:.2f}")

    # 2) MOMENT load encoder & embedding extraction
    moment = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    moment.init()
    moment.to(device).float()

    train_emb, train_labels = get_embedding(moment, train_loader, device=device)
    test_emb, test_labels = get_embedding(moment, test_loader, device=device)

    # 3) per-patient epoch length computation
    train_lengths, test_lengths = compute_lengths_per_patient(
        x_root=args.x_root,
        split_pickle_path=args.split_pickle,
    )

    total_epochs_tr = sum(train_lengths)
    total_epochs_te = sum(test_lengths)

    k = args.k
    assert len(train_emb) == total_epochs_tr * k, "train embeddings length != epochs*k"
    assert len(test_emb) == total_epochs_te * k, "test embeddings length != epochs*k"
    assert len(train_labels) == total_epochs_tr * k
    assert len(test_labels) == total_epochs_te * k

    # 4) 7 sub-epochs → 1 epoch embedding
    Xn_train, yn_train, train_counts_folded = fold_k_by_counts(
        train_emb,
        train_labels,
        train_lengths,
        k=k,
        reduce="mean",
        labels_are_per_epoch=False,
        label_mode="mode",
        strict=True,
    )
    Xn_test, yn_test, test_counts_folded = fold_k_by_counts(
        test_emb,
        test_labels,
        test_lengths,
        k=k,
        reduce="mean",
        labels_are_per_epoch=False,
        label_mode="mode",
        strict=True,
    )

    # 5) night-level dataloaders
    n = len(Xn_train)
    cut = max(1, int(0.9 * n))

    dl_tr = DataLoader(
        Nights(Xn_train[:cut], yn_train[:cut]),
        batch_size=args.batch_size_night,
        shuffle=True,
        collate_fn=collate_nights,
    )
    dl_val = DataLoader(
        Nights(Xn_train[cut:], yn_train[cut:]),
        batch_size=args.batch_size_night,
        shuffle=False,
        collate_fn=collate_nights,
    )
    dl_te = DataLoader(
        Nights(Xn_test, yn_test),
        batch_size=args.batch_size_night,
        shuffle=False,
        collate_fn=collate_nights,
    )

    # 6) Mamba-based sequence model
    model = SeqHead(
        in_dim=1024,
        n_classes=5,
        d_model=args.d_model,
        depth=args.depth,
        bidi=args.bidi,
        p_drop=0.1,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_metric = "acc"
    mode = "max" if use_metric == "acc" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode=mode,
        factor=0.3,
        patience=5,
        min_lr=3e-5,
    )

    best_state = None
    best_metric = -float("inf") if use_metric == "acc" else float("inf")
    patience_es = args.patience_es
    since_improve = 0

    # 7) Training loop
    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tot_loss, tot_tokens = 0.0, 0

        for xb, yb, mb in dl_tr:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            out = model(xb, mb)
            loss = masked_ce(out, yb, mb, label_smooth=args.label_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item() * mb.sum().item()
            tot_tokens += mb.sum().item()

        train_loss = tot_loss / max(1, tot_tokens)

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

                val_c += (pred[mask] == yb[mask]).sum().item()
                val_t += mask.sum().item()
                vloss = masked_ce(logits, yb, mb, label_smooth=args.label_smooth)
                val_loss_sum += vloss.item() * mb.sum().item()

                all_preds.extend(pred[mask].cpu().numpy())
                all_trues.extend(yb[mask].cpu().numpy())

        val_acc = val_c / max(1, val_t)
        val_loss = val_loss_sum / max(1, val_t)
        val_macro_f1 = f1_score(all_trues, all_preds, average="macro")
        cur_lr = opt.param_groups[0]["lr"]

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

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    test_acc = (all_preds == all_trues).mean()
    test_macro_f1 = f1_score(all_trues, all_preds, average="macro")
    print(f"Test acc: {test_acc:.4f} | Test macro-F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
