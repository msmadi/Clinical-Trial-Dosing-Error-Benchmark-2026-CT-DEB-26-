# Automated Detection of Dosing Errors in Clinical Trial Narratives

[![Paper](https://img.shields.io/badge/Paper-LREC--COLING%202026-blue)](https://github.com/msmadi/Clinical-Trial-Dosing-Error-Benchmark-2026-CT-DEB-26-)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Automated Detection of Dosing Errors in Clinical Trial Narratives: A Multi-Modal Feature Engineering Approach with LightGBM"** (LREC-COLING 2026).

> **Abstract:** We present an automated system for detecting dosing errors in clinical trial narratives using gradient boosting with comprehensive multi-modal feature engineering. On the CT-DEB benchmark (42,112 narratives, ~5% positive rate), we achieve **0.8725 test ROC-AUC** through 5-fold ensemble averaging with Optuna-optimized hyperparameters. Our approach combines 3,451 features spanning traditional NLP, dense semantic embeddings, domain-specific medical patterns, and transformer scores, demonstrating that sparse lexical features remain highly competitive with dense representations for specialized clinical text classification under severe class imbalance.

---

## 📂 Repository Organization

> **Important**: All Python code is located in the **`code/`** directory. Navigate there before running any scripts:
> ```bash
> cd code
> python complete_pipeline_with_error_analysis.py
> ```

---

## 📋 Table of Contents

- [Key Results](#-key-results)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Reproduction](#-reproduction)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Key Results

| Metric | Score | Note |
|--------|-------|------|
| **Test ROC-AUC** | **0.8725** | 5-fold ensemble |
| **Cross-Validation AUC** | **0.8833 ± 0.0091** | Stable performance |
| **Recall @ threshold=0.3744** | **26.1%** | F1-optimized |
| **Recall @ threshold=0.10** | **60.0%** | Safety-critical mode |
| **Precision @ threshold=0.3744** | **33.9%** | Manageable FP rate |
| **Feature Efficiency** | **100.10%** | With only 100/3451 features |


### Feature Importance Distribution

| Category | Features | Total Gain (%) | Avg Gain |
|----------|----------|----------------|----------|
| Word/Char (TF-IDF) | 3,020 | 62.38 | 30.40 |
| Sentence Embeddings | 386 | 37.07 | 141.37 |
| Medical Patterns | 43 | 0.49 | 16.84 |
| Transformer Scores | 2 | 0.06 | 43.90 |

### Ablation Study Impact

Removing **sentence embeddings** causes the largest performance degradation (**-2.39%**), while removing word/char features impacts only **-0.25%**, demonstrating their complementary nature.

---

## ✨ Features

- **🔥 Optuna-Optimized Pipeline**: Automated hyperparameter tuning with 50 trials
- **🎯 5-Fold Ensemble**: Robust cross-validation with stable predictions
- **📊 Comprehensive Feature Engineering**: 
  - Traditional NLP (TF-IDF, character n-grams)
  - Dense semantic embeddings (all-MiniLM-L6-v2)
  - Domain-specific medical patterns
  - Transformer probability scores (BiomedBERT, DeBERTa-v3)
- **⚡ Feature Efficiency Analysis**: Identifies optimal feature subset
- **🔍 Systematic Ablation Studies**: Quantifies contribution of each feature category
- **📈 Threshold Optimization**: Adjustable recall-precision trade-offs
- **📉 Visualization Tools**: ROC curves, PR curves, feature importance plots
- **💾 Memory Efficient**: Sparse matrix storage with NPZ compression

---

## 🚀 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/msmadi/Clinical-Trial-Dosing-Error-Benchmark-2026-CT-DEB-26-.git
cd Clinical-Trial-Dosing-Error-Benchmark-2026-CT-DEB-26-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: All Python scripts are located in the `code/` directory.

### Dependencies

```txt
# Core ML/NLP
lightgbm==4.1.0
optuna==3.4.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4

# Deep Learning & Transformers
torch==2.1.1
transformers==4.35.2
sentence-transformers==2.2.2

# Data Loading
datasets==2.15.0

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Utilities
tqdm==4.66.1
```

---


---

## 📊 Dataset

We use the **CT-DEB (Clinical Trial Dosing Error Benchmark)** dataset:

```python
from datasets import load_dataset

# Load dataset from Hugging Face
ds_train = load_dataset("sssohrab/ct-dosing-errors-benchmark", split="train")
ds_val = load_dataset("sssohrab/ct-dosing-errors-benchmark", split="validation")
ds_test = load_dataset("sssohrab/ct-dosing-errors-benchmark", split="test")
```

### Dataset Statistics

| Split | Total | Negative | Positive | % Positive |
|-------|-------|----------|----------|------------|
| Train | 29,478 | 28,126 | 1,352 | 4.6% |
| Validation | 6,316 | 6,031 | 285 | 4.5% |
| Test | 6,318 | 6,008 | 310 | 4.9% |
| **Total** | **42,112** | **40,165** | **1,947** | **4.6%** |

**Text Characteristics:**
- Mean length: 387 characters (std: 312)
- Median length: 298 characters
- Range: 47 to 5,847 characters

---


---

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{alsmadi2026dosing,
  title={Automated Detection of Dosing Errors in Clinical Trial Narratives: A Multi-Modal Feature Engineering Approach with LightGBM},
  author={Mohammad, AL-Smadi},
  booktitle={Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)},
  year={2026},
  note={Under review}
}
```

**Dataset Citation:**

```bibtex
@misc{sohrab2024ctdeb,
  title={CT-DEB: Clinical Trial Dosing Error Benchmark},
  author={Sohrab, Soheila and others},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/sssohrab/ct-dosing-errors-benchmark}}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- 🔧 Additional feature engineering methods
- 🎯 Alternative model architectures (e.g., deep learning baselines)
- 📊 Additional evaluation metrics and visualizations
- 🌍 Multi-lingual support
- 🏥 Integration with clinical trial management systems

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---



<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for safer clinical trials

</div>
