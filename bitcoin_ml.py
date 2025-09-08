import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "data", "bitcoin.csv")
PLOT_DIR = os.path.join(ROOT, "outputs", "plots")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    # Load
    df = pd.read_csv(DATA_PATH)
    print("Loaded shape:", df.shape)

    # Basic EDA plots
    plt.figure(figsize=(15, 5))
    plt.plot(df['Close'])
    plt.title('Bitcoin Close price.')
    plt.ylabel('Price in dollars.')
    save_fig(os.path.join(PLOT_DIR, "close_series.png"))

    # Drop Adj Close if identical to Close
    if "Adj Close" in df.columns and (df["Close"] == df["Adj Close"]).all():
        df = df.drop(columns=["Adj Close"])

    # Nulls
    nulls = df.isnull().sum()
    nulls.to_csv(os.path.join(OUT_DIR, "null_counts.csv"))

    # Distribution & boxplots for OHLC
    features = ['Open', 'High', 'Low', 'Close']
    for col in features:
        plt.figure(figsize=(8,4))
        sn.distplot(df[col])
        plt.title(f"Distribution - {col}")
        save_fig(os.path.join(PLOT_DIR, f"dist_{col}.png"))

        plt.figure(figsize=(8,2.5))
        sn.boxplot(df[col], orient='h')
        plt.title(f"Boxplot - {col}")
        save_fig(os.path.join(PLOT_DIR, f"box_{col}.png"))

    # Feature engineering
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year.astype(int)
    df['month'] = df['Date'].dt.month.astype(int)
    df['day'] = df['Date'].dt.day.astype(int)
    data_grouped = df.groupby('year').mean(numeric_only=True)

    # Year-wise barplots
    for col in ['Open', 'High', 'Low', 'Close']:
        plt.figure(figsize=(10,4))
        data_grouped[col].plot.bar()
        plt.title(f"Year-wise mean {col}")
        save_fig(os.path.join(PLOT_DIR, f"year_mean_{col}.png"))

    df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
    df['open-close']  = df['Open'] - df['Close']
    df['low-high']  = df['Low'] - df['High']
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna().reset_index(drop=True)

    # Target balance pie
    plt.figure(figsize=(4,4))
    plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
    plt.title("Target balance")
    save_fig(os.path.join(PLOT_DIR, "target_balance.png"))

    # Correlation heatmap (threshold>0.9)
    plt.figure(figsize=(8,8))
    sn.heatmap(df.corr(numeric_only=True) > 0.9, annot=True, cbar=False)
    plt.title("Highly correlated features (>|0.9|)")
    save_fig(os.path.join(PLOT_DIR, "corr_over_0_9.png"))

    # Modeling
    features = df[['open-close', 'low-high', 'is_quarter_end']]
    target = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X_scaled, target, test_size=0.3, random_state=42, stratify=target)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("SVC_poly", SVC(kernel='poly', probability=True)),
        ("XGBClassifier", XGBClassifier(eval_metric="logloss", use_label_encoder=False, n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0))
    ]

    lines = []
    for name, model in models:
        model.fit(X_train, Y_train)
        train_auc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])
        valid_auc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:,1])
        lines.append(f"{name}: train_AUC={train_auc:.4f}, valid_AUC={valid_auc:.4f}")
        print(lines[-1])

    # Choose LR for robustness (lower overfitting)
    best_name, best_model = models[0]
    # Save confusion matrix for LR
    ConfusionMatrixDisplay.from_estimator(best_model, X_valid, Y_valid, cmap='Blues')
    plt.title("Confusion Matrix - Logistic Regression (valid)")
    save_fig(os.path.join(PLOT_DIR, "confusion_matrix_lr.png"))

    # Save artifacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "logreg_model.joblib"))
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write(f"class_balance: {dict(df['target'].value_counts())}\n")

    print("Done. Plots in outputs/plots, models saved, metrics written.")

if __name__ == "__main__":
    main()
