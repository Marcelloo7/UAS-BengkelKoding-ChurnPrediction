# ============================================================
# src/preprocessing.py
# Preprocessing + Modeling - UAS Churn Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PATH, PLOTS_DIR, MODELS_DIR, TARGET_COL, COLS_DROP,
                    COLS_KATEGORIKAL, COLS_NUMERIK, COLS_SCALE,
                    TEST_SIZE, RANDOM_STATE)

plt.style.use('seaborn-v0_8-whitegrid')


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[Prep] Dataset dimuat: {df.shape[0]} baris")
    return df


def hapus_fitur_tidak_relevan(df):
    cols = [c for c in COLS_DROP if c in df.columns]
    df = df.drop(columns=cols)
    print(f"[Prep] Kolom dihapus: {cols}")
    return df


def tangani_missing_value(df):
    print("\n[Prep] Menangani Missing Value...")

    # Tangani inconsistent data: age negatif tidak masuk akal -> treat sebagai missing
    n_age_negatif = (df['age'] < 0).sum()
    if n_age_negatif > 0:
        df.loc[df['age'] < 0, 'age'] = np.nan
        print(f"  age negatif ditemukan: {n_age_negatif} baris -> diubah jadi NaN")

    # age -> imputasi median (distribusi skewed)
    imp_median = SimpleImputer(strategy='median')
    df['age'] = imp_median.fit_transform(df[['age']])

    # total_spent -> imputasi median
    df['total_spent'] = imp_median.fit_transform(df[['total_spent']])

    # satisfaction_score -> imputasi median
    df['satisfaction_score'] = imp_median.fit_transform(df[['satisfaction_score']])

    # gender -> imputasi modus (kategorikal)
    imp_modus = SimpleImputer(strategy='most_frequent')
    df['gender'] = imp_modus.fit_transform(df[['gender']]).ravel()

    # coupon_code -> drop kolom (40.9% missing, tidak informatif)
    if 'coupon_code' in df.columns:
        df = df.drop(columns=['coupon_code'])
        print("  coupon_code dihapus (missing value 40.9%)")

    print(f"  Total missing value tersisa: {df.isnull().sum().sum()}")
    return df


def tangani_duplikasi(df):
    dup = df.duplicated().sum()
    print(f"\n[Prep] Duplikasi: {dup} baris")
    if dup > 0:
        df = df.drop_duplicates()
        print(f"  Duplikasi dihapus. Sisa: {len(df)} baris")
    return df


def tangani_outlier(df):
    print("\n[Prep] Menangani Outlier (Metode IQR)...")
    cols_outlier = ['total_spent', 'avg_order_value', 'lifetime_value',
                    'marketing_spend_per_user', 'avg_session_time']
    before = len(df)
    for col in cols_outlier:
        if col in df.columns:
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df  = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    print(f"  Baris sebelum: {before} | Setelah: {len(df)} | Dihapus: {before-len(df)}")
    return df


def encoding(df):
    print("\n[Prep] Encoding Fitur Kategorikal...")
    le_dict = {}
    cat_cols = [c for c in COLS_KATEGORIKAL if c in df.columns]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        print(f"  {col}: {list(le.classes_)}")
    return df, le_dict


def split_data(df):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[Prep] Split: Train={X_train.shape[0]} | Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def scaling(X_train, X_test):
    print("\n[Prep] Scaling dengan StandardScaler...")
    cols = [c for c in COLS_SCALE if c in X_train.columns]
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols]  = scaler.transform(X_test[cols])
    print(f"  Kolom di-scale: {cols}")
    return X_train, X_test, scaler


def handling_imbalance(X_train, y_train):
    print("\n[Prep] Handling Imbalance dengan SMOTE...")
    print(f"  Sebelum: {dict(pd.Series(y_train).value_counts())}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Setelah: {dict(pd.Series(y_res).value_counts())}")

    # Visualisasi
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    before = pd.Series(y_train).value_counts()
    after  = pd.Series(y_res).value_counts()
    axes[0].bar(['Tidak Churn', 'Churn'], before.sort_index().values,
                color=['#378ADD', '#D85A30'])
    axes[0].set_title('Sebelum SMOTE')
    for i, v in enumerate(before.sort_index().values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    axes[1].bar(['Tidak Churn', 'Churn'], after.sort_index().values,
                color=['#378ADD', '#D85A30'])
    axes[1].set_title('Setelah SMOTE')
    for i, v in enumerate(after.sort_index().values):
        axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.suptitle('Distribusi Label: Sebelum vs Setelah SMOTE', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'prep_smote.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: prep_smote.png")
    return X_res, y_res


def evaluasi_model(nama, y_test, y_pred, warna='Blues'):
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
    plt.title(f'Confusion Matrix - {nama} (Prep)')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    fname = f"prep_{nama.lower().replace(' ', '_')}_cm.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.show()
    return {'Model': nama, 'Akurasi': acc, 'Presisi': prec,
            'Recall': rec, 'F1 Score': f1}


def run_preprocessing():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  MODELING DENGAN PREPROCESSING")
    print("="*60)

    df = load_data()
    df = hapus_fitur_tidak_relevan(df)
    df = tangani_missing_value(df)
    df = tangani_duplikasi(df)
    df = tangani_outlier(df)
    df, le_dict = encoding(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scaling(X_train, X_test)
    X_train_sm, y_train_sm  = handling_imbalance(X_train, y_train)

    hasil = []

    # Model 1: Logistic Regression
    print("\n[1/3] Melatih Logistic Regression (Preprocessing)...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_sm, y_train_sm)
    hasil.append(evaluasi_model('Logistic Regression', y_test,
                                lr.predict(X_test), 'Blues'))

    # Model 2: Random Forest
    print("\n[2/3] Melatih Random Forest (Preprocessing)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train_sm, y_train_sm)
    hasil.append(evaluasi_model('Random Forest', y_test,
                                rf.predict(X_test), 'Greens'))

    # Model 3: Voting Classifier
    print("\n[3/3] Melatih Voting Classifier (Preprocessing)...")
    voting = VotingClassifier(estimators=[
        ('lr',  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ], voting='soft')
    voting.fit(X_train_sm, y_train_sm)
    hasil.append(evaluasi_model('Voting Classifier', y_test,
                                voting.predict(X_test), 'Oranges'))

    df_hasil = pd.DataFrame(hasil).set_index('Model')
    print("\n" + "="*60)
    print("  RANGKUMAN MODELING DENGAN PREPROCESSING")
    print("="*60)
    print(df_hasil.round(4).to_string())
    print("\n[Preprocessing] Selesai!")

    # Simpan scaler dan label encoder
    import joblib
    joblib.dump(scaler,  os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(le_dict, os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    print("[Model] Scaler dan Label Encoder disimpan!")

    return df_hasil, X_train_sm, X_test, y_train_sm, y_test, rf, scaler, le_dict


if __name__ == "__main__":
    run_preprocessing()