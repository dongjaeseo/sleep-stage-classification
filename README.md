# Sleep Stage Classification using MOMENT Foundation Model

This project explores sleep stage classification using the **MOMENT Time Series Foundation Model** on **PhysioNet PSG data**, and compares its performance with **SleepFM**, a model specialized for sleep staging.  
This work is ongoing as part of my research internship at Seoul National University.

---

## 1. Overview

**MOMENT** is a large-scale pretrained time-series foundation model capable of handling:
- Forecasting  
- Classification  
- Imputation  
- Anomaly detection  

It is pretrained on large-scale time-series datasets that include biosignals such as EEG and ECG.

This project examines:
- How a general-purpose foundation model performs on sleep staging  
- Impact of channel count (1 / 3 / 5 channels)  
- Classifier comparison (Linear vs SVM)  
- Frozen vs fine-tuned encoder  
- Sequence-level modeling using **Mamba**  

---

## 2. MOMENT Model Summary

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

Each 512-sample chunk is passed through MOMENT.  
During inference, the 7 logits are averaged to classify the full epoch.

---

## 4. Experiments

### 4.1 Channel Configurations
Experiments were run with:
- **1 channel**
- **3 channels**
- **5 channels**

**Best performance:**  
3 channels (C3-M2, E1-M2, Chin1–Chin2)

---

### 4.2 Classifier Comparison
Two classifiers were tested:
- **Linear classifier**
- **SVM**

**Result:**  
The linear classifier outperformed SVM.

---

### 4.3 Frozen vs Fine-Tuned Encoder

Two modes were tested:
- **Frozen encoder (linear probe)**
- **Fine-tuned encoder**

**Best MOMENT result:**  
- **Macro-F1 = 0.5999**  
- Configuration: 3 channels + linear classifier + trainable encoder

---

### 4.4 Comparison with SleepFM
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
