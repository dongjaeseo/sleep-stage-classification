
'''
ch_idx:
  Channel selections used in experiments:
    - 1 ch  : [C3-M2]
    - 3 ch  : [C3-M2, E1-M2, Chin1-Chin2]
    - 5 ch  : [C3-M2, E1-M2, Chin1-Chin2, Airflow, ECG]

probe_mode:
  Select classifier type:
    'linear' → simple linear probe
    'SVM'    → GridSearch SVC (sklearn)

freeze:
  Controls freezing/unfreezing of embedder and encoder blocks.

use_subset:
  Enable this for running quick tests on a smaller data subset.
'''


EXPERIMENTS = {
    "single_linear": dict(
        ch_idx=[2],                  # 1ch
        probe_mode="linear",
        freeze=True,
        use_subset=False,
        epoch=10,
        train_subset=50000,
        test_subset=10000,
        seed=42,
    ),
    "single_svm": dict(
        ch_idx=[2],
        probe_mode="SVM",
        freeze=True,
        use_subset=False,
        epoch=10,
        train_subset=50000,
        test_subset=10000,
        seed=42,
    ),
    "multi_linear": dict(
        ch_idx=[2, 6, 7],            # 3ch
        probe_mode="linear",
        freeze=True,
        use_subset=False,
        epoch=10,
        train_subset=50000,
        test_subset=10000,
        seed=42,
    ),
    "multi_svm": dict(
        ch_idx=[2, 6, 7],
        probe_mode="SVM",
        freeze=True,
        use_subset=False,
        epoch=10,
        train_subset=50000,
        test_subset=10000,
        seed=42,
    ),
    "multi_linear_unfreeze": dict(
        ch_idx=[2, 6, 7],
        probe_mode="linear",
        freeze=False,                # different here
        use_subset=False,
        epoch=10,
        train_subset=50000,
        test_subset=10000,
        seed=42,
    ),
    "multi_linear_5ch": dict(
        ch_idx=[2, 6, 7, 10, 12],    # 5ch
        probe_mode="linear",
        freeze=True,
        use_subset=False,
        epoch=10,
        train_subset=50000,
        test_subset=10000,
        seed=42,
    ),
}