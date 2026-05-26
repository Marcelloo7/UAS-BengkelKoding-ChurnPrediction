# ============================================================
# src/direct_modeling.py
# Direct Modeling (tanpa preprocessing) - UAS Churn Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PATH, PLOTS_DIR, MODELS_DIR, TARGET_COL,
                    COLS_DROP, TEST_SIZE, RANDOM_STATE)

plt.style.use('seaborn-v0_8-whitegrid')


def load_and_prepare(df=None):
    """Load data dan encoding minimal untuk direct modeling."""
    if df is None:
        df = pd.read_csv(DATA_PATH)

    df_d = df.copy()

    # Hapus kolom tidak relevan
    cols_to_drop = [c for c in COLS_DROP if c in df_d.columns]
    df_d = df_d.drop(columns=cols_to_drop)

    # Drop missing value (minimal preprocessing)
    df_d = df_d.dropna()

    # Label Encoding semua kolom object
    le = LabelEncoder()
    for col in df_d.select_dtypes(include='object').columns:
        df_d[col] = le.fit_transform(df_d[col].astype(str))

    X = df_d.drop(TARGET_COL, axis=1)
    y = df_d[TARGET_COL]

    print(f"[Direct] Data siap: {X.shape[0]} baris, {X.shape[1]} fitur")
    print(f"[Direct] Distribusi target: {dict(y.value_counts())}")
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[Direct] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def evaluasi_model(nama, y_test, y_pred, warna='Blues'):
    """Cetak metrik evaluasi dan tampilkan confusion matrix."""
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  EVALUASI: {nama}")
    print(f"{'='*50}")
    print(f"  Akurasi  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Presisi  : {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Tidak Churn','Churn'])}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap=warna,
                xticklabels=['Tidak Churn', 'Churn'],
                yticklabels=['Tidak Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {nama}')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    fname = f"direct_{nama.lower().replace(' ', '_')}_cm.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.show()
    print(f"[Plot] Tersimpan: {fname}")

    return {'Model': nama, 'Akurasi': acc, 'Presisi': prec,
            'Recall': rec, 'F1 Score': f1}


def run_direct_modeling(df=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  DIRECT MODELING (Tanpa Preprocessing)")
    print("="*60)

    X, y = load_and_prepare(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    hasil = []

    # --------------------------------------------------
    # Model 1: Logistic Regression (Konvensional)
    # --------------------------------------------------
    print("\n[1/3] Melatih Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    hasil.append(evaluasi_model('Logistic Regression',
                                y_test, lr.predict(X_test), 'Blues'))

    # --------------------------------------------------
    # Model 2: Random Forest (Ensemble Bagging)
    # --------------------------------------------------
    print("\n[2/3] Melatih Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    hasil.append(evaluasi_model('Random Forest',
                                y_test, rf.predict(X_test), 'Greens'))

    # --------------------------------------------------
    # Model 3: Voting Classifier (Ensemble Voting)
    # LR + SVM + KNN
    # --------------------------------------------------
    print("\n[3/3] Melatih Voting Classifier (LR + SVM + KNN)...")
    voting = VotingClassifier(estimators=[
        ('lr',  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ], voting='soft')
    voting.fit(X_train, y_train)
    hasil.append(evaluasi_model('Voting Classifier',
                                y_test, voting.predict(X_test), 'Oranges'))

    # --------------------------------------------------
    # Tabel Perbandingan
    # --------------------------------------------------
    df_hasil = pd.DataFrame(hasil).set_index('Model')
    print("\n" + "="*60)
    print("  RANGKUMAN DIRECT MODELING")
    print("="*60)
    print(df_hasil.round(4).to_string())

    # Visualisasi perbandingan
    metrics = ['Akurasi', 'Presisi', 'Recall', 'F1 Score']
    x  = np.arange(len(metrics))
    w  = 0.25
    colors = ['#378ADD', '#1D9E75', '#D85A30']

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (idx, row) in enumerate(df_hasil.iterrows()):
        bars = ax.bar(x + i*w, row[metrics].values, w,
                      label=idx, color=colors[i], edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}',
                    ha='center', fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_title('Perbandingan Performa - Direct Modeling')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'direct_comparison.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: direct_comparison.png")
    print("\n[Direct Modeling] Selesai!")

    return df_hasil, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_direct_modeling()
