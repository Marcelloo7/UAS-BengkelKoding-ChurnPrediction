# ============================================================
# src/tuning.py
# Hyperparameter Tuning & Feature Selection - UAS Churn Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOTS_DIR, MODELS_DIR, RANDOM_STATE

plt.style.use('seaborn-v0_8-whitegrid')


def feature_importance(rf_model, feature_names):
    print("\n" + "="*60)
    print("  FEATURE IMPORTANCE")
    print("="*60)
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    plt.figure(figsize=(13, 5))
    plt.bar(range(len(importances)),
            importances[indices],
            color='#5DCAA5', edgecolor='white')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right', fontsize=8)
    plt.title('Feature Importance - Random Forest')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'tuning_feature_importance.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: tuning_feature_importance.png")

    print("\nTop 10 Fitur Paling Berpengaruh:")
    for i in range(min(10, len(importances))):
        print(f"  {i+1:2d}. {feature_names[indices[i]]:<35} {importances[indices[i]]:.4f}")
    return indices


def evaluasi_model(nama, y_test, y_pred, warna='Blues'):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  EVALUASI: {nama} (Setelah Tuning)")
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
    plt.title(f'Confusion Matrix - {nama} (Tuned)')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    fname = f"tuning_{nama.lower().replace(' ', '_')}_cm.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.show()
    return {'Model': f'{nama} (Tuned)', 'Akurasi': acc,
            'Presisi': prec, 'Recall': rec, 'F1 Score': f1}


def run_tuning(X_train, X_test, y_train, y_test,
               rf_before=None, hasil_prep=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING & FEATURE SELECTION")
    print("="*60)

    feature_names = X_train.columns.tolist()

    # Feature Importance (gunakan RF default dulu jika tidak ada)
    if rf_before is None:
        rf_before = RandomForestClassifier(n_estimators=100,
                                           random_state=RANDOM_STATE)
        rf_before.fit(X_train, y_train)
    feature_importance(rf_before, feature_names)

    hasil = []

    # --------------------------------------------------
    # Tuning 1: Logistic Regression
    # --------------------------------------------------
    print("\n[1/3] Tuning Logistic Regression...")
    param_lr = {
        'C'      : [0.01, 0.1, 1, 10, 100],
        'solver' : ['lbfgs', 'liblinear'],
        'penalty': ['l1', 'l2']
    }
    rs_lr = RandomizedSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_distributions=param_lr,
        n_iter=10, cv=5, scoring='f1',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rs_lr.fit(X_train, y_train)
    print(f"  Best params: {rs_lr.best_params_}")
    print(f"  Best CV F1 : {rs_lr.best_score_:.4f}")
    hasil.append(evaluasi_model('Logistic Regression', y_test,
                                rs_lr.best_estimator_.predict(X_test), 'Blues'))

    # --------------------------------------------------
    # Tuning 2: Random Forest
    # --------------------------------------------------
    print("\n[2/3] Tuning Random Forest...")
    param_rf = {
        'n_estimators'     : [50, 100, 200],
        'max_depth'        : [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'max_features'     : ['sqrt', 'log2']
    }
    rs_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_rf,
        n_iter=15, cv=5, scoring='f1',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rs_rf.fit(X_train, y_train)
    print(f"  Best params: {rs_rf.best_params_}")
    print(f"  Best CV F1 : {rs_rf.best_score_:.4f}")
    hasil.append(evaluasi_model('Random Forest', y_test,
                                rs_rf.best_estimator_.predict(X_test), 'Greens'))

    # --------------------------------------------------
    # Tuning 3: Voting Classifier
    # Gunakan best estimator dari LR dan RF
    # --------------------------------------------------
    print("\n[3/3] Tuning Voting Classifier...")
    voting_tuned = VotingClassifier(estimators=[
        ('lr',  rs_lr.best_estimator_),
        ('rf',  rs_rf.best_estimator_),
        ('knn', KNeighborsClassifier(n_neighbors=3))
    ], voting='soft')
    voting_tuned.fit(X_train, y_train)
    hasil.append(evaluasi_model('Voting Classifier', y_test,
                                voting_tuned.predict(X_test), 'Oranges'))

    # --------------------------------------------------
    # Tabel perbandingan 9 model (3 skenario x 3 model)
    # --------------------------------------------------
    df_tuned = pd.DataFrame(hasil).set_index('Model')
    print("\n" + "="*60)
    print("  RANGKUMAN HYPERPARAMETER TUNING")
    print("="*60)
    print(df_tuned.round(4).to_string())

    # Gabungkan dengan hasil sebelumnya jika ada
    if hasil_prep is not None:
        df_all = pd.concat([hasil_prep, df_tuned])
        print("\n" + "="*60)
        print("  PERBANDINGAN SEMUA 9 MODEL")
        print("="*60)
        print(df_all.round(4).to_string())
        plot_perbandingan_semua(df_all)

    # Simpan model terbaik (Random Forest tuned)
    best_model = rs_rf.best_estimator_
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n[Model] Model terbaik disimpan: {model_path}")

    # Simpan scaler (akan dibutuhkan deployment)
    return df_tuned, best_model


def plot_perbandingan_semua(df_all):
    """Visualisasi perbandingan semua 9 model."""
    metrics = ['Akurasi', 'Presisi', 'Recall', 'F1 Score']
    colors  = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        vals   = df_all[metric].values
        labels = df_all.index.tolist()
        bars   = axes[i].bar(range(len(vals)), vals,
                             color=colors[:len(vals)], edgecolor='white')
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=25,
                                ha='right', fontsize=8)
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1.15)
        for bar in bars:
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.005,
                         f'{bar.get_height():.3f}',
                         ha='center', fontsize=7)

    plt.suptitle('Perbandingan Semua Model (9 Kombinasi)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'tuning_all_comparison.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: tuning_all_comparison.png")


if __name__ == "__main__":
    print("Jalankan melalui main.py")
