# German Credit Data MLOps Project

## Overview
This project implements a Machine Learning pipeline for the **German Credit Data** dataset, with a focus on **MLOps best practices**. The goal is to predict credit risk (good or bad credit) while maintaining a robust, scalable, and reproducible ML workflow.

---

## Dataset
- Source: [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- Features: 20 attributes including credit history, employment status, purpose, and more.
- Target: `CreditRisk` (Good / Bad)
- Data is split into training and testing sets as part of the pipeline.

---

## Project Structure
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks for EDA and model experimentation
├── src/ # Source code
│ ├── data_preprocessing.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
├── models/ # Saved models and artifacts
├── requirements.txt # Python dependencies
├── README.md
└── config.yaml # Configuration for data paths, model parameters, etc.
## Features
- **Data preprocessing**: Missing value handling, encoding categorical variables, scaling numerical features.
- **Modeling**: Supports multiple ML algorithms (Logistic Regression, Random Forest, XGBoost).
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score, and ROC-AUC.
- **MLOps Integration**:
  - Config-driven pipeline using YAML
  - Versioned models and artifacts
  - Reproducible experiments
  - Easy deployment-ready structure

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone <repository_url>
cd german_credit_data
