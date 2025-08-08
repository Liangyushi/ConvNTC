# ConvNTC: convolutional neural tensor completion for detecting “A–A–B” type biological triplets
ConvNTC consists of three parts: the dataset module, the multilinear relationship learning module, and the nonlinear relationship learning module. ConvNTC is primarily applied to two prediction tasks i.e. miRNA–miRNA–disease and drug–drug–cell triplets. Taking “miRNA–miRNA–disease” as an example, the general outline of ConvNTC is shown as follow：

<img width="3442" height="2546" alt="framework_1018" src="https://github.com/user-attachments/assets/3634836f-2a23-4aad-b26f-72b2baf36c27" />

---

# Runing Environment
Python version: 3.10.13

PyTorch version: 2.0.1

TensorLy version: 0.8.1

NumPy version: 1.26.4

CUDA version: 11.7, NVIDIA GeForce RTX 4090

---

## 📁 Project Structure

```
├── data/
│   ├── NCI_ddi5cv/                 # NCI dataset with 5-fold cross-validation
│   │   ├── information/            # Metadata and supporting files
│   │   ├── Simcell_cosine.csv      # Cell similarity matrix
│   │   ├── newdrugsim_nci.csv      # Drug similarity matrix
│   │   ├── neg_mmd_*.txt           # Negative samples (MMD)
│   │   ├── pos_mmd_*.txt           # Positive samples (MMD)
│   │....
│   ├── hmdvd32_neg/                # Negative samples for hmdvd32
│   ├── oneil_ddi5cv/               # Oneil dataset 5-fold
│
├── experiments/
│   ├── NCI_compare/
│   │   ├── ConvNTC_nci.py
│   │   └── MCTD_nci.py
│   ├── Oneil_compare/
│   │   ├── ConvNTC_oneil.py
│   │   └── MCTD_oneil.py
│   ├── hmdvd32_compare/
│       ├── ConvNTC_hmdvd32.py
│       └── MCTD_hmdvd32.py
│
├── src/
│   ├── ConvKAN/                    # Main model implementations
│   │   ├── ConvNTC.py              # Convolutional neural tensor model
│   │   └── MCTD.py                 # Multi-channel tensor decomposition model
│   ├── compareLinearModels.py     # Comparison with linear models
│   ├── compareNonlinearModels.py  # Comparison with nonlinear models
│   ├── drug_nci_data.py           # Data loader for NCI
│   ├── drug_oneil_data.py         # Data loader for Oneil
│   └── hmdvd32_data.py            # Data loader for hmdvd32
```

---

## 🚀 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run an experiment**
   ```bash
   python experiments/NCI_compare/ConvNTC_nci.py
   ```

3. **Model options**
   - ConvNTC: Convolutional Neural Tensor Completion
   - MCTD: Multi-channel Tensor Decomposition

---

## 📁 Data Description

All datasets are organized for 5-fold cross-validation, with matched positive/negative interaction pairs and similarity matrices. Below is a detailed breakdown:

### 📦 `data/NCI_ddi5cv/`

- **information/**
  - `allDrugsmile_nci.csv`: SMILES strings for drugs in NCI dataset.
  - `nci_new.rda`, `nci_smile.ipynb`, `nci_split.R`: R script and notebooks for data preprocessing and splitting.
  - `Simcell_cosine.csv`: Cosine similarity between cell lines.
  - `newdrugsim_nci.csv`: Drug–drug similarity matrix.

- **Main Files:**
  - `pos_mmd_1neg_*.txt`: Positive drug-cell interaction samples.
  - `neg_mmd_1neg_*.txt`: Negative samples for corresponding folds.

---

### 📦 `data/hmdvd32_neg/`

- **Subfolders (`1n`, `2n`, ..., `10n`)**: Contain synthetic negative samples with different ratios.
  - `dsSim.csv`, `dtSim.csv`, `meshSim.csv`, `msSim.csv`, `mtSim.csv`, `seqSim.csv`: Various similarity measures (disease, drug, sequence, etc.)
  - `pos_mmd_1neg_*.txt`, `neg_mmd_1neg_*.txt`: Positive and negative interaction files for 5 folds.

- **information/**
  - `disease.txt`, `miRNA.txt`: Node lists.
  - `pos_dm.txt`, `pos_mm.txt`, `pos_mmd.txt`: Different edge types for miRNA–disease, miRNA–miRNA, and miRNA–miRNA–disease graphs.

---

### 📦 `data/oneil_ddi5cv/`

- **information/**
  - `Simcell_cosine.csv`, `Simdrug_cosine.csv`: Cosine similarities between cell lines and drugs.

- **Main Files:**
  - `cell_feature.csv`, `drug_feature.csv`: Feature matrices for each entity.
  - `newdrugsim.csv`: Drug similarity matrix.
  - `pos_mmd_1neg_*.txt`, `neg_mmd_1neg_*.txt`: 5-fold split positive and negative samples.

---

Each dataset is structured to support consistent cross-validation and evaluation with the same format.

---

## 🧪 Experiments

Each dataset has its own set of scripts for comparison:

- **NCI_compare/**
  - `ConvNTC_nci.py`: Runs the ConvNTC model on the NCI dataset.
  - `MCTD_nci.py`: Runs the MCTD model on the NCI dataset.

- **Oneil_compare/**
  - `ConvNTC_oneil.py`: Runs the ConvNTC model on the Oneil dataset.
  - `MCTD_oneil.py`: Runs the MCTD model on the Oneil dataset.

- **hmdvd32_compare/**
  - `ConvNTC_hmdvd32.py`: Runs the ConvNTC model on the hmdvd32 dataset.
  - `MCTD_hmdvd32.py`: Runs the MCTD model on the hmdvd32 dataset.

Each script loads data, trains the model, and evaluates performance metrics like AUC, RMSE, etc.

---

## 🧠 Source Code Overview (`src/`)

- **ConvKAN/**
  - `ConvNTC.py`: Implementation of the Convolutional Neural Tensor Completion model.
  - `MCTD.py`: Implementation of Multi-Channel Tensor Decomposition model.

- **compareLinearModels.py**: Baseline comparison using traditional linear models (e.g., Ridge, Logistic Regression).

- **compareNonlinearModels.py**: Baseline comparison using non-linear models (e.g., Random Forest, MLP).

- **drug_nci_data.py**: Data loading utilities for the NCI dataset.
- **drug_oneil_data.py**: Data loading utilities for the Oneil dataset.
- **hmdvd32_data.py**: Data loading utilities for the hmdvd32 dataset.

- **utils.py**: Common utility functions used across training and evaluation.

---

# Citation
Pei Liu, Xiao Liang, Yue Li, Jiawei Luo, ConvNTC: convolutional neural tensor completion for detecting “A–A–B” type biological triplets, Briefings in Bioinformatics, Volume 26, Issue 4, July 2025, bbaf372, https://doi.org/10.1093/bib/bbaf372

---


