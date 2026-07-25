# 🛡️ FraudGuard AI — Credit Card Fraud Detection

A production-ready machine learning pipeline for real-time credit card fraud detection, featuring a multi-model benchmark suite and an interactive Streamlit dashboard.

---

## Overview

Credit card fraud causes billions in annual losses through direct chargebacks, customer friction from false positives, and regulatory penalties. This project builds an intelligent ML system that balances catching fraud while minimizing disruption to legitimate cardholders.

---

## Dataset

The dataset contains anonymized European credit card transactions from a 48-hour window. Due to PCI-DSS compliance requirements, raw transaction fields are transformed into PCA components. The dataset exhibits extreme class imbalance, with fraud accounting for less than 0.2% of all transactions.

---

## Approach

1. **Preprocessing** — RobustScaler applied to skewed features to handle extreme outliers
2. **Class Imbalance** — SMOTE oversampling applied exclusively to the training split to prevent data leakage
3. **Model Benchmarking** — Five architectures trained and compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, and a Deep Neural Network
4. **Metric Focus** — Precision-Recall AUC prioritized over ROC-AUC due to class imbalance
5. **Threshold Tuning** — Decision threshold optimized for maximum financial savings rather than default 0.50

**XGBoost** was selected as the production champion based on its superior PR-AUC, inference speed, and overall financial ROI.

---

## Project Structure

```
MDD_CC/
├── Credit_Card_Fraud_Detection_Simple.ipynb   # Main ML pipeline
├── build_professional_mdd.py                  # MDD document generator
├── MDD_CreditCardFraud_Professional.docx      # Model Design Document
└── README.md
---

## Dashboard

The trained model powers **FraudGuard AI**, an interactive Streamlit operations center built for fraud analysts.

- **Live Batch Scoring** — Upload a CSV of transactions and get instant fraud predictions with risk scores
- **PCA Visualization** — Interactive scatter plot of the latent feature space, with normal and fraudulent transactions highlighted in distinct colors for quick visual triage
- **Analyst Review Panel** — Drill into individual flagged transactions, inspect feature values, view a risk gauge, and issue manual Approve or Decline decisions that update in real time
