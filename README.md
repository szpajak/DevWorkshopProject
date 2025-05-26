
# Development Workshop Project

**Explainable AI for Citation Screening**

## Project Overview

This project aims to develop and evaluate explainable AI models for automating citation screening in systematic reviews. The solution leverages NLP and deep learning techniques to classify the relevance of scientific articles, with a focus on transparency and interpretability.

## Directory Structure

```
.
├── data/                # Datasets (raw and preprocessed)
│   ├── raw/             # Original data sources
│   └── preprocessed/    # Cleaned and processed datasets
├── models/              # Trained model files
├── reports/             # Experiment reports
├── shap/                # SHAP explainability outputs
├── src/                 # Source code
│   ├── models/          # Model training and evaluation notebooks
│   ├── preprocessing/   # Data preprocessing scripts and notebooks
│   └── ui/              # Main app file with helper functions
├── .gitignore
├── README.md
```

## Main Components

- **Data**: Contains both raw and preprocessed datasets for training and evaluation.
- **src/models**: Jupyter notebooks for model development, training, and evaluation (e.g., `model.ipynb`).
- **src/preprocessing**: Scripts and notebooks for data cleaning and preparation (e.g., `preprocess_utils.py`).
- **models**: Stores trained PyTorch model files (e.g., `CD011975_model.pt`).
- **shap**: Stores SHAP values and explainability artifacts.
- **reports**: Contains experiment reports and analysis in both PDF and Jupyter notebook formats.