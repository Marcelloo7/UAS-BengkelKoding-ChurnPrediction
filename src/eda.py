# ============================================================
# src/eda.py
# Exploratory Data Analysis - UAS Churn Prediction
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PATH, PLOTS_DIR, TARGET_COL,
                    COLS_NUMERIK, COLS_KATEGORIKAL, COLS_BINER)

plt.style.use('seaborn-v0_8-whitegrid')


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[EDA] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def eksplorasi_awal(df):
    print("\n" + "="*60)
    print("  1. EKSPLORASI AWAL")
    print("="*60)
    print("\n--- 5 Baris Pertama ---")
    print(df.head())
    print("\n--- Info Dataset ---")
    df.info()
    print("\n--- Statistik Deskriptif ---")
    print(df.describe())
    print(f"\nFitur Kategorikal ({len(COLS_KATEGORIKAL)}): {COLS_KATEGORIKAL}")
    print(f"Fitur Biner       ({len(COLS_BINER)})      : {COLS_BINER}")
    print(f"Fitur Numerik     ({len(COLS_NUMERIK)})     : {COLS_NUMERIK}")


def cek_missing_value(df):
    print("\n" + "="*60)
    print("  2. MISSING VALUE")
    print("="*60)
    missing     = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    mv_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing (%)': missing_pct.round(2)
    })
    mv_ada = mv_df[mv_df['Missing Count'] > 0]
    print(mv_ada.to_string())

    # Visualisasi
    missing_only = missing[missing > 0]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(missing_only.index, missing_only.values,
                   color='#FF6B6B', edgecolor='white')
    for bar, val in zip(bars, missing_only.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=9)
    plt.title('Jumlah Missing Value per Kolom')
    plt.ylabel('Jumlah Missing Value')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_missing_value.png'), dpi=150)
    plt.show()
    print(f"\n[Plot] Tersimpan: eda_missing_value.png")

    print("\nInsight:")
    print("  - coupon_code: 40.9% missing -> akan dihapus pada preprocessing")
    print("  - age, total_spent, gender, satisfaction_score: akan diimputasi")
    return mv_df


def distribusi_target(df):
    print("\n" + "="*60)
    print("  3. DISTRIBUSI TARGET (CHURN)")
    print("="*60)
    churn_count = df[TARGET_COL].value_counts()
    churn_pct   = df[TARGET_COL].value_counts(normalize=True) * 100
    print(f"  0 (Tidak Churn) : {churn_count[0]} ({churn_pct[0]:.1f}%)")
    print(f"  1 (Churn)       : {churn_count[1]} ({churn_pct[1]:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bars = ax1.bar(['Tidak Churn (0)', 'Churn (1)'],
                   churn_count.values,
                   color=['#378ADD', '#D85A30'], edgecolor='white')
    ax1.set_title('Distribusi Churn')
    ax1.set_ylabel('Jumlah Pelanggan')
    for bar, val in zip(bars, churn_count.values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50, str(val),
                 ha='center', fontsize=11, fontweight='bold')

    ax2.pie(churn_count.values,
            labels=['Tidak Churn', 'Churn'],
            autopct='%1.1f%%',
            colors=['#378ADD', '#D85A30'],
            startangle=90, explode=[0, 0.05])
    ax2.set_title('Proporsi Churn')

    plt.suptitle('Distribusi Variabel Target: Churn', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_target_churn.png'), dpi=150)
    plt.show()
    print(f"[Plot] Tersimpan: eda_target_churn.png")

    print("\nInsight - Imbalance Class:")
    print("  - Data tidak seimbang: 84.7% tidak churn vs 15.3% churn")
    print("  - Perlu penanganan imbalance pada tahap preprocessing")


def distribusi_kategorikal(df):
    print("\n" + "="*60)
    print("  4. DISTRIBUSI FITUR KATEGORIKAL")
    print("="*60)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(COLS_KATEGORIKAL):
        order = df[col].value_counts().index
        sns.countplot(data=df, y=col, ax=axes[i], order=order, palette='Blues_d')
        axes[i].set_title(f'Distribusi: {col}')
        axes[i].set_xlabel('Jumlah')
    plt.suptitle('Distribusi Fitur Kategorikal', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_kategorikal.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: eda_kategorikal.png")

    # Churn rate per kategori
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(COLS_KATEGORIKAL):
        churn_rate = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
        churn_rate.plot.bar(ax=axes[i], color='#D85A30', edgecolor='white')
        axes[i].set_title(f'Churn Rate: {col}')
        axes[i].set_ylabel('Churn Rate')
        axes[i].set_ylim(0, 0.4)
        axes[i].tick_params(axis='x', rotation=30)
    plt.suptitle('Churn Rate per Kategori', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_churnrate_kategorikal.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: eda_churnrate_kategorikal.png")


def distribusi_numerik(df):
    print("\n" + "="*60)
    print("  5. DISTRIBUSI FITUR NUMERIK")
    print("="*60)
    n_cols = 3
    n_rows = (len(COLS_NUMERIK) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()
    for i, col in enumerate(COLS_NUMERIK):
        sns.histplot(df[col].dropna(), ax=axes[i], kde=True,
                     color='#378ADD', bins=30)
        axes[i].set_title(col)
        axes[i].set_xlabel('')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Distribusi Fitur Numerik', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_numerik.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: eda_numerik.png")


def analisis_outlier(df):
    print("\n" + "="*60)
    print("  6. ANALISIS OUTLIER")
    print("="*60)
    n_cols = 3
    n_rows = (len(COLS_NUMERIK) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()
    for i, col in enumerate(COLS_NUMERIK):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color='#FF6B6B')
        axes[i].set_title(f'Boxplot: {col}')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Deteksi Outlier - Fitur Numerik', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_outlier.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: eda_outlier.png")

    print("\nJumlah Outlier per Kolom (Metode IQR):")
    for col in COLS_NUMERIK:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        n   = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)].shape[0]
        print(f"  {col}: {n} outlier ({n/len(df)*100:.1f}%)")


def heatmap_korelasi(df):
    print("\n" + "="*60)
    print("  7. HEATMAP KORELASI")
    print("="*60)
    df_num = df[COLS_NUMERIK + COLS_BINER + [TARGET_COL]].copy()
    plt.figure(figsize=(18, 12))
    sns.heatmap(df_num.corr(), annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=0.5,
                annot_kws={'size': 7}, center=0)
    plt.title('Heatmap Korelasi Fitur Numerik', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_korelasi.png'), dpi=150)
    plt.show()
    print("[Plot] Tersimpan: eda_korelasi.png")

    churn_corr = df_num.corr()[TARGET_COL].drop(TARGET_COL)\
                       .sort_values(key=abs, ascending=False)
    print("\nTop 10 Korelasi dengan Churn:")
    print(churn_corr.head(10).to_string())


def run_eda():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_data()
    eksplorasi_awal(df)
    cek_missing_value(df)
    distribusi_target(df)
    distribusi_kategorikal(df)
    distribusi_numerik(df)
    analisis_outlier(df)
    heatmap_korelasi(df)
    print("\n[EDA] Selesai! Semua plot tersimpan di folder output/plots/")
    return df


if __name__ == "__main__":
    run_eda()
