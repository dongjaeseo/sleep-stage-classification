import pickle, os
import glob, re
from tqdm import tqdm
import numpy as np

# train: 710144 test: 179367 sleep epochs
'''
Purpose of this codepage:
From preprocessed PhysioNet data, 
create memmap files
x_train (train#, 13, 7680)
x_test (test#, 13, 7680)
y_train (train#,)
y_test (test#)

where x contains physiological signals of sleep epochs of patients
y contains label of specific sleep epoch

Experiment files of this repository ASSUMES that you have 
train_x.npy / test_x.npy / train_y.npy / test_y.npy
'''

'''
1. train/test split according to SleepFM
file_path should lead to the file containing information about patient labeled as train / valid / test
train_list / test_list should contain list of patient_id
'''
file_path = "/ssd/kdpark/sleepfm-codebase/physionet_final/dataset.pickle"
with open(file_path, "rb") as f:
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


# 2. Count total number of samples
EPOCH_RE = re.compile(r"_(\d+)\.npy$")
# Ideally x data format should be (13, 7680) according to PhysioNet
C, L = 13, 7680

n_train, n_test = 0, 0
'''
Files under x_root should be {patient_id}/{patient_id}_{epoch_number}.npy with data shape (13,7680)
Files under y_root should be {patient_id}.pickle which contains whole epoch label for one patient
'''
x_root = "/ssd/kdpark/sleepfm-codebase/physionet_final/X"
y_root = "/ssd/kdpark/sleepfm-codebase/physionet_final/Y"

pdirs = sorted(d for d in glob.glob(os.path.join(x_root, "*")) if os.path.isdir(d))

for pdir in tqdm(pdirs):
    pid = os.path.basename(pdir)
    if not os.path.exists(os.path.join(y_root, f"{pid}.pickle")):
        continue
    x_files = sorted(glob.glob(os.path.join(pdir, f"{pid}_*.npy")))
    if not x_files:
        continue

    if pid in train_set:
        n_train += len(x_files)
    elif pid in test_set:
        n_test += len(x_files)

print("train:", n_train, "test:", n_test)

# 3. memmap file generation
train_X = np.memmap("/ssd/kdpark/dongjae/moment/moment_sleep/train_X.npy", dtype="float32", mode="w+", shape=(n_train, C, L))
train_y = np.memmap("/ssd/kdpark/dongjae/moment/moment_sleep/train_y.npy", dtype="int64", mode="w+", shape=(n_train,))
test_X  = np.memmap("/ssd/kdpark/dongjae/moment/moment_sleep/test_X.npy",  dtype="float32", mode="w+", shape=(n_test, C, L))
test_y  = np.memmap("/ssd/kdpark/dongjae/moment/moment_sleep/test_y.npy",  dtype="int64", mode="w+", shape=(n_test,))

# 4. Data fill in memmap file
i_train, i_test = 0, 0

def extract_label(v):
    if isinstance(v, dict) and len(v) > 0:
        v = next(iter(v.values()))
    return int(v)

for pdir in tqdm(pdirs):
    pid = os.path.basename(pdir)
    y_path = os.path.join(y_root, f"{pid}.pickle")
    if not os.path.exists(y_path):
        continue

    with open(y_path, "rb") as f:
        yraw = pickle.load(f)

    key2label = {}
    for k, v in yraw.items():
        try:
            key2label[str(k)[:-4]] = extract_label(v)
        except Exception:
            continue

    x_files = sorted(glob.glob(os.path.join(pdir, f"{pid}_*.npy")))
    if not x_files:
        continue

    for xf in x_files:
        m = EPOCH_RE.search(os.path.basename(xf))
        if not m:
            continue
        ei = int(m.group(1))
        key = f"{pid}_{ei}"
        if key not in key2label:
            continue

        arr = np.load(xf, allow_pickle=False)  # Check data shape == (13,7680)
        if arr.shape != (C, L):
            raise ValueError(f"Shape mismatch {arr.shape} in {xf}")

        if arr.shape == (L, C):  
            arr = arr.T
        assert arr.shape == (C, L), f"Shape mismatch {arr.shape} in {xf}"

        y = key2label[key]
        if pid in train_set:
            train_X[i_train] = arr.astype("float32")
            train_y[i_train] = y
            i_train += 1
        elif pid in test_set:
            test_X[i_test] = arr.astype("float32")
            test_y[i_test] = y
            i_test += 1

print("filled train:", i_train, "expected:", n_train)
print("filled test:", i_test, "expected:", n_test)
assert i_train == n_train, "train number mismatch"
assert i_test == n_test, "test number mismatch"
# flush + close
train_X.flush(); train_y.flush()
test_X.flush(); test_y.flush()
del train_X, train_y, test_X, test_y
