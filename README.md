# Sleep Stage Classification using MOMENT Foundation Model

---
## Folder Structure
'''
sleep-stage-classification/
├── preprocessing/
│   └── 1_save_data.py              # Preprocess raw PSG and save epoch-level data
│
├── experiments/
│   ├── 2_single_linear.py          # 1-channel → linear classifier baseline
│   ├── 3_single_SVM.py             # 1-channel → SVM baseline
│   ├── 4_multi_linear.py           # 3-channel → linear classifier
│   ├── 5_multi_SVM.py              # 3-channel → SVM classifier
│   ├── 6_multi_linear_unfreeze.py  # Multi-channel classifier with encoder unfreezing
│   ├── 7_multi_linear_5ch.py       # 5-channel linear classifier variant
│   └── 8_mamba_whole_night.py      # Whole-night sequence refinement using Mamba
│
├── MOMENT on PhysioNet.pptx        # Slides explaining the baseline & results
├── requirements.txt                # Python dependencies
└── README.md                       # Project description
'''
## 1. Overview

**How well can a general-purpose time-series foundation model understand sleep?**  
This project evaluates **MOMENT** [(Goswami et al., 2024)](https://arxiv.org/abs/2402.03885), a large-scale pretrained foundation model for physiological time-series data, on the classical sleep-stage classification task and compares its performance with **SleepFM** [(Thapa et al., 2024)](https://arxiv.org/abs/2405.17766), a sleep-specialized foundation model. The goal is to understand what factors most influence downstream performance.

This study examines:

- **Channel sensitivity:** performance differences with 1 / 3 / 5 biosignal channels  
- **Classifier choices:** Linear vs. SVM on frozen embeddings  
- **Fine-Tuning strategy:** frozen encoder vs. full fine-tuning  
- **Sequence modeling:** adding a bidirectional **Mamba** encoder [(Gu & Dao, 2024)](https://arxiv.org/abs/2312.00752) to capture long-range temporal structure  
- **Key finding:** Mamba-based sequence learning consistently boosts macro-F1, even with minimal fine-tuning

This repository includes preprocessing, embedding extraction, sequence modeling, and training scripts

---

## 2. MOMENT Model Summary
<img width="1019" height="763" alt="image" src="https://github.com/user-attachments/assets/38fbd86b-03c2-4fbc-8107-fe07bad52327" />
### Pretraining Strategy
- **Masking:** Random segments of the input sequence are masked  
- **Patching:** Signals are split into fixed-length patches  
- **Encoding:** A Transformer encoder learns temporal dependencies  
- **Reconstruction:** Masked patches are restored during pretraining  

### Classification
- Transformer embeddings contain learned time-series features  
- A classifier head is trained for sleep staging  
- MOMENT is **univariate**, processing one channel at a time  
- For multichannel PSG, embeddings are extracted per channel and averaged  

### Input Length
- MOMENT processes a **fixed sequence length of 512**

---

## 3. Data: PhysioNet PSG

- **Sampling rate:** 256 Hz  
- **Epoch length:** 30 seconds  
- **Raw sequence length:** 7680 samples  
- **Downsampling (stride 2):** 7680 → 3840  
- **Crop:** Remove 1 second from start and end  
- **Remaining:** 28 seconds = 3584 samples  
- **Split:** 3584 / 7 = **7 chunks (each 512 samples)**
- <img width="983" height="217" alt="image" src="https://github.com/user-attachments/assets/abd15651-191a-480b-bb48-eecdceecbe22" />


Each 512-sample chunk is passed through MOMENT.  
During inference, the 7 logits are averaged to classify the full epoch.

---

## 4. Experiments

### 4.1 Channel Configurations
Experiments were run with:
- **1 channel**
- **3 channels**
- **5 channels**
<img width="689" height="271" alt="image" src="https://github.com/user-attachments/assets/7e490023-a332-4902-a0ad-f85acd4212a9" />

**Best performance:**  
3 channels (C3-M2, E1-M2, Chin1–Chin2)

---

### 4.2 Classifier Comparison
Two classifiers were tested:
- **Linear classifier**
- **SVM**
<img width="690" height="300" alt="image" src="https://github.com/user-attachments/assets/d8b7b1be-1ceb-42b0-b333-4763bccf5417" />

**Result:**  
The linear classifier outperformed SVM.

---

### 4.3 Frozen vs Fine-Tuned Encoder

Two modes were tested:
- **Frozen encoder (linear probe)**
- **Fine-tuned encoder**
<img width="689" height="198" alt="image" src="https://github.com/user-attachments/assets/ceec2855-d5bc-42f5-a8f3-51ea5a3d4d9f" />

**Best MOMENT result:**  
- **Macro-F1 = 0.5999**  
- Configuration: 3 channels + linear classifier + trainable encoder

---

### 4.4 Comparison with SleepFM
<img width="1021" height="320" alt="image" src="https://github.com/user-attachments/assets/ed69382a-7a8f-486d-b5a9-22506d10cae6" />
- SleepFM achieved a higher macro-F1  
- MOMENT still showed meaningful performance despite being a general-purpose model

---

## 5. Sequence Modeling with Mamba

To capture temporal continuity across the night, MOMENT embeddings for all epochs were passed into a **Mamba** sequence model.


### 5.1 Mamba Results

To model long-range transitions across the night, MOMENT embeddings for each epoch were fed into a Mamba sequence model.

Two evaluation setups were tested:

#### • Full-night sequence  
- **Test Accuracy:** 0.7876  
- **Macro-F1:** 0.7566  

Using the entire night preserves true sleep architecture (N1→N2→N3→REM cycles), and Mamba handled these transitions well. This setup gave the highest improvement, especially for stages that depend on context such as **N1** and **REM**.

#### • Short 10-epoch window  
- **Test Accuracy:** 0.7466  
- **Macro-F1:** 0.7103  

With only 10 consecutive epochs, the model had much less temporal information, and performance dropped accordingly. Even then, it still outperformed the chunk-based MOMENT classifier, showing that transition modeling is beneficial even in shorter windows.

### Summary
Mamba provided a clear performance boost over independent epoch classification.  
Sequence modeling reduced noisy predictions, improved stage consistency, and moved the macro-F1 into a more reliable range.

| Model                      | Macro-F1 | Notes                 |
|---------------------------|----------|------------------------|
| MOMENT (best config)      | 0.5999   | 3-ch, linear, finetune |
| Mamba (10-epoch window)   | 0.7103   | sequence modeling      |
| Mamba (full night)        | 0.7566   | best overall           |

---

## 6. Conclusion

- General foundation model (MOMENT) performs competitively on sleep staging  
- Best configuration: **3 channels + linear classifier + fine-tuning**  
- **Mamba** significantly improves sequence-level performance  

## References

- MOMENT: Time-Series Foundation Model (ICML 2024)  
- PhysioNet Dataset  
- SleepFM: Sleep Stage Classification Model  
- Mamba: Selective State Space Model
