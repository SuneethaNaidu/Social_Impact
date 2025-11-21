# txn_categorizer — Automated Transaction Categorisation

## Overview
This project demonstrates a reproducible pipeline to train and serve a transaction categoriser.
Key features:
- TF-IDF + Logistic Regression baseline (fast, explainable)
- Configurable taxonomy via YAML
- Evaluation script producing macro / per-class F1 and confusion matrix
- Simple Flask inference API

## How to run
1. Install dependencies:
