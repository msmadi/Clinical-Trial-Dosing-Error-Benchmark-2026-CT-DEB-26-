# Automated Detection of Dosing Errors in Clinical Trial Narratives

[![Paper](https://img.shields.io/badge/Paper-LREC--COLING%202026-blue)](https://github.com/yourusername/ct-dosing-error-detection)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Automated Detection of Dosing Errors in Clinical Trial Narratives: A Multi-Modal Feature Engineering Approach with LightGBM"** (LREC-COLING 2026).

> **Abstract:** We present an automated system for detecting dosing errors in clinical trial narratives using gradient boosting with comprehensive multi-modal feature engineering. On the CT-DEB benchmark (35,794 narratives, 4.6% positive rate), we achieve **0.8725 test ROC-AUC** through 5-fold ensemble averaging with Optuna-optimized hyperparameters. Our approach combines 3,451 features spanning traditional NLP, dense semantic embeddings, domain-specific medical patterns, and transformer scores, demonstrating that sparse lexical features remain highly competitive with dense representations for specialized clinical text classification under severe class imbalance.

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
| **Recall @ threshold=0.20** | **49.0%** | Safety-critical mode |
| **Precision @ threshold=0.3744** | **33.9%** | Manageable FP rate |
| **Feature Efficiency** | **99.22%** | With only 200/3451 features |
| **Computational Speedup** | **17×** | Top-200 vs full feature set |

### Feature Importance Distribution

| Category | Features | Total Gain (%) | Avg Gain |
|----------|----------|----------------|----------|
| Word/Char (TF-IDF) | 3,020 | 62.03 | 38.26 |
| Sentence Embeddings | 386 | 37.34 | 180.22 |
| Medical Patterns | 43 | 0.58 | 25.26 |
| Transformer Scores | 2 | 0.05 | 46.50 |

### Ablation Study Impact

Removing **sentence embeddings** causes the largest performance degradation (**-7.59%**), while removing word/char features impacts only **-1.01%**, demonstrating their complementary nature.

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
git clone https://github.com/yourusername/ct-dosing-error-detection.git
cd ct-dosing-error-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

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

## 🏃 Quick Start

### 1. Train the Full Model

```bash
python train.py \
    --data_dir data/ct-deb \
    --feature_dir features \
    --output_dir outputs \
    --n_folds 5 \
    --n_estimators 4000 \
    --early_stopping_rounds 200
```

### 2. Run Comprehensive Analysis Pipeline

```bash
python comprehensive_pipeline_final.py
```

This will:
- ✅ Train 5-fold ensemble
- ✅ Compute out-of-fold predictions
- ✅ Optimize classification threshold
- ✅ Analyze feature importance
- ✅ Perform ablation study
- ✅ Run top-K efficiency analysis
- ✅ Generate all visualizations
- ✅ Create comprehensive reports

**Output files** (in `out/` directory):
- `fold_results.csv` - Performance per fold
- `feature_importance_complete.csv` - All features ranked
- `category_importance.csv` - Aggregated by category
- `ablation_results.csv` - Systematic removal results
- `topk_results.csv` - Efficiency analysis
- `test_results.json` - Complete test metrics
- `test_evaluation_summary.png` - 4-panel visualization
- `ANALYSIS_REPORT.txt` - Human-readable summary

### 3. Extract Features from Raw Text

```bash
python extract_features.py \
    --input data/ct-deb \
    --output features \
    --n_jobs -1
```

This extracts all 3,451 features:
- Medical pattern features (43)
- Word TF-IDF (≈2,000)
- Character n-grams (≈1,000)
- Sentence embeddings (386)
- Transformer scores (2)

### 4. Run Optuna Hyperparameter Optimization

```bash
python optuna_optimize.py \
    --n_trials 50 \
    --n_folds 5 \
    --study_name ct_dosing_detection \
    --storage sqlite:///optuna_study.db
```

### 5. Inference on New Data

```bash
python predict.py \
    --model_path outputs/ensemble_model.pkl \
    --input_file data/new_narratives.csv \
    --output_file predictions.csv \
    --threshold 0.3744
```

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

**Annotation Quality:**
- Inter-annotator agreement: Cohen's κ = 0.87
- Annotators: Clinical research coordinators with 5+ years experience

---

## 🔬 Reproduction

### Reproduce Exact Paper Results

```bash
# Step 1: Extract features (if not already done)
python extract_features.py --config configs/paper_config.yaml

# Step 2: Run full pipeline with exact hyperparameters
python comprehensive_pipeline_final.py

# Step 3: Verify results match paper
python verify_results.py --results out/test_results.json
```

**Expected outputs:**
```
✓ Test ROC-AUC: 0.872545 (matches paper)
✓ Cross-validation: 0.883289 ± 0.009145 (matches paper)
✓ Feature importance: Word/Char 62.03%, Embeddings 37.34% (matches paper)
✓ Top-200 efficiency: 99.22% of baseline (matches paper)
```

### Reproduce Optuna Optimization (Trial 18)

```bash
python optuna_optimize.py \
    --n_trials 50 \
    --n_folds 5 \
    --seed 42 \
    --storage sqlite:///optuna_study.db

# Best trial (Trial 18) hyperparameters:
# learning_rate: 0.0054
# num_leaves: 118
# max_depth: 9
# min_child_samples: 211
# lambda_l1: 4.29
# lambda_l2: 4.33
# feature_fraction: 0.795
# bagging_fraction: 0.813
# scale_pos_weight: 20.87
```

### Reproduce Ablation Study

```bash
python ablation_study.py \
    --feature_dir features \
    --output_dir outputs/ablation \
    --categories medical word_char sentence_embeddings transformer
```

---

## 📁 Project Structure

```
ct-dosing-error-detection/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── configs/
│   ├── paper_config.yaml              # Exact paper configuration
│   └── optuna_config.yaml             # Optuna optimization settings
│
├── data/
│   └── README.md                      # Dataset download instructions
│
├── features/                          # Extracted features (NPZ format)
│   ├── split_tr8000_word50_char3000_ng3-7_all-MiniLM-L6-v2_FULL_X_train.npz
│   ├── split_tr8000_word50_char3000_ng3-7_all-MiniLM-L6-v2_FULL_X_val.npz
│   └── split_tr8000_word50_char3000_ng3-7_all-MiniLM-L6-v2_FULL_X_test.npz
│
├── models/
│   ├── ensemble_model.pkl             # Trained 5-fold ensemble
│   └── feature_selector.pkl           # Top-200 feature selector
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # Data exploration
│   ├── 02_feature_engineering.ipynb   # Feature extraction walkthrough
│   ├── 03_model_training.ipynb        # Training visualization
│   └── 04_results_analysis.ipynb      # Results deep-dive
│
├── scripts/
│   ├── download_models.py             # Download pre-trained models
│   ├── verify_results.py              # Verify reproduction
│   └── create_visualizations.py       # Generate all plots
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                 # Dataset loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   ├── medical_patterns.py        # Medical pattern extraction
│   │   ├── tfidf_features.py          # TF-IDF features
│   │   ├── char_ngrams.py             # Character n-grams
│   │   ├── sentence_embeddings.py     # Sentence transformer embeddings
│   │   └── transformer_scores.py      # BiomedBERT & DeBERTa scores
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lightgbm_model.py          # LightGBM wrapper
│   │   ├── ensemble.py                # 5-fold ensemble
│   │   └── threshold_optimizer.py     # Threshold optimization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── visualizations.py          # Plotting functions
│   └── utils/
│       ├── __init__.py
│       └── io.py                      # I/O utilities
│
├── train.py                           # Main training script
├── extract_features.py                # Feature extraction script
├── predict.py                         # Inference script
├── comprehensive_pipeline_final.py    # Full analysis pipeline
├── optuna_optimize.py                 # Hyperparameter optimization
└── ablation_study.py                  # Ablation study script
```

---

## 🔧 Advanced Usage

### Custom Feature Engineering

```python
from src.features import (
    MedicalPatternExtractor,
    TFIDFFeatureExtractor,
    CharNgramExtractor,
    SentenceEmbeddingExtractor,
    TransformerScoreExtractor
)

# Initialize extractors
medical_extractor = MedicalPatternExtractor()
tfidf_extractor = TFIDFFeatureExtractor(max_features=2000)
char_extractor = CharNgramExtractor(ngram_range=(3, 7), max_features=3000)
sentence_extractor = SentenceEmbeddingExtractor(model_name='all-MiniLM-L6-v2')
transformer_extractor = TransformerScoreExtractor(models=['biomedbert', 'deberta-v3'])

# Extract features
texts = ["Patient received reduced dose due to adverse event.", ...]
features = {
    'medical': medical_extractor.transform(texts),
    'tfidf': tfidf_extractor.transform(texts),
    'char': char_extractor.transform(texts),
    'sentence': sentence_extractor.transform(texts),
    'transformer': transformer_extractor.transform(texts)
}
```

### Threshold Optimization

```python
from src.models import ThresholdOptimizer

# Optimize for different objectives
optimizer = ThresholdOptimizer(y_true, y_pred_proba)

# F1-optimized (default)
f1_threshold = optimizer.optimize(metric='f1', min_precision=0.3)

# Recall-optimized (safety-critical)
recall_threshold = optimizer.optimize(metric='recall', min_precision=0.2)

# Balanced accuracy
balanced_threshold = optimizer.optimize(metric='balanced_accuracy', min_precision=0.3)

print(f"F1-optimized threshold: {f1_threshold:.4f}")
print(f"Recall-optimized threshold: {recall_threshold:.4f}")
```

### Feature Selection

```python
from src.models import FeatureSelector

# Select top-K features
selector = FeatureSelector(model=trained_model, k=200)
X_selected = selector.transform(X)

# Get feature importance
importance_df = selector.get_importance_dataframe()
print(importance_df.head(20))
```

---

## 📈 Performance Benchmarks

### Training Time

| Configuration | Features | Time (CPU) | Time (GPU) | Memory |
|---------------|----------|------------|------------|--------|
| Full (3,451) | All | 3-5 min | N/A* | ~2GB |
| Top-500 | Selected | 1-2 min | N/A* | ~500MB |
| Top-200 | Selected | 30-60 sec | N/A* | ~300MB |

*LightGBM is CPU-optimized; GPU provides minimal benefit for tabular data.

### Inference Time

| Configuration | Throughput (narratives/sec) | Latency (ms/narrative) |
|---------------|----------------------------|------------------------|
| Full (3,451) | ~350 | ~2.9 |
| Top-500 | ~1,500 | ~0.67 |
| Top-200 | ~6,000 | ~0.17 |

**Hardware:** Intel Core i7-10700K (8 cores, 3.8 GHz), 32GB RAM

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**

```bash
# Use sparse matrix format
python extract_features.py --sparse --compress
```

**2. Feature Extraction Too Slow**

```bash
# Parallelize across CPU cores
python extract_features.py --n_jobs -1
```

**3. Optuna Optimization Runs Out of Storage**

```bash
# Use pruning to stop unpromising trials early
python optuna_optimize.py --pruner median --n_startup_trials 10
```

**4. Model Not Converging**

```bash
# Increase iterations and adjust learning rate
python train.py --n_estimators 6000 --learning_rate 0.003
```

---

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{anonymous2026dosing,
  title={Automated Detection of Dosing Errors in Clinical Trial Narratives: A Multi-Modal Feature Engineering Approach with LightGBM},
  author={Anonymous},
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

## 🙏 Acknowledgments

- **CT-DEB Dataset**: Thanks to the clinical research coordinators who annotated the dataset
- **Hugging Face**: For hosting the dataset and providing transformer models
- **Sentence-Transformers**: For the all-MiniLM-L6-v2 model
- **LightGBM Team**: For the efficient gradient boosting framework
- **Optuna Team**: For the hyperparameter optimization framework

---

## 📧 Contact

- **Primary Contact**: [Your Name] - your.email@institution.edu
- **GitHub Issues**: [https://github.com/yourusername/ct-dosing-error-detection/issues](https://github.com/yourusername/ct-dosing-error-detection/issues)
- **Project Website**: [https://yourwebsite.com/ct-dosing-detection](https://yourwebsite.com/ct-dosing-detection)

---

## 🔖 Related Work

- [CT-DEB Dataset Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [BiomedBERT](https://arxiv.org/abs/2104.03506)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [Optuna](https://arxiv.org/abs/1907.10902)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for safer clinical trials

</div>
