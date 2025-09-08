# Bitcoin Price Prediction using Machine Learning in Python

Machine learning proves immensely helpful in many industries in automating tasks that earlier required human labor; one such application is predicting whether a particular **buy signal** will be profitable or not.

This repository implements:
- Data loading and EDA (OHLC data)
- Feature engineering (open–close, low–high, quarter-end flag)
- Model training (Logistic Regression, SVM, XGBoost) to predict next-day up/down signal
- Evaluation via ROC-AUC and confusion matrix
- Reproducible notebook and Python script
- Saved plots and model artifacts

> **Note:** A small **sample dataset** is included at `data/bitcoin.csv`. Replace it with your real dataset having columns: `Date, Open, High, Low, Close, Adj Close, Volume` for authentic results.

## Project Structure
```
bitcoin-price-prediction-ml/
├── data/
│   └── bitcoin.csv
├── images/
│   ├── banner.png
│   └── pipeline.png
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── models/
│   └── plots/
├── src/
│   └── bitcoin_ml.py
├── requirements.txt
└── README.md
```

## Quickstart
```bash
# 1) Create & activate environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the pipeline
python src/bitcoin_ml.py

# 4) (Optional) Open the notebook
jupyter notebook notebooks/analysis.ipynb
```

## Replacing the Dataset
- Put your real file at `data/bitcoin.csv` with columns:
  `Date, Open, High, Low, Close, Adj Close, Volume`.
- Date format preferred: `YYYY-MM-DD`.
- The script will automatically generate plots in `outputs/plots/` and a confusion matrix.

## Outputs
- `outputs/plots/`: EDA charts and confusion matrix
- `outputs/models/`: Saved scaler and trained Logistic Regression model
- `outputs/metrics.txt`: ROC-AUC (train/valid) and class balance

## Credits
This codebase follows the outline you provided and is productionized for GitHub.
