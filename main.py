# ============================================================
# main.py
# Runner utama - UAS Bengkel Koding Data Science
# Churn Prediction - Sales & Marketing Customer
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.eda            import run_eda
from src.direct_modeling import run_direct_modeling
from src.preprocessing  import run_preprocessing
from src.tuning         import run_tuning


def print_header(judul):
    print("\n")
    print("█"*60)
    print(f"  {judul}")
    print("█"*60)


def main():
    print_header("UAS BENGKEL KODING DATA SCIENCE")
    print("  Churn Prediction - Sales & Marketing Customer")
    print("  Dataset: 15.000 records | Target: churn (0/1)")

    # --------------------------------------------------
    # BAGIAN 1: EDA
    # --------------------------------------------------
    print_header("BAGIAN 1: EXPLORATORY DATA ANALYSIS")
    run_eda()

    # --------------------------------------------------
    # BAGIAN 2: DIRECT MODELING
    # --------------------------------------------------
    print_header("BAGIAN 2: DIRECT MODELING")
    hasil_direct, _, _, _, _ = run_direct_modeling()

    # --------------------------------------------------
    # BAGIAN 3: PREPROCESSING + MODELING
    # --------------------------------------------------
    print_header("BAGIAN 3: MODELING DENGAN PREPROCESSING")
    hasil_prep, X_train, X_test, y_train, y_test, rf_model, scaler, le_dict = run_preprocessing()

    # --------------------------------------------------
    # BAGIAN 4: HYPERPARAMETER TUNING
    # --------------------------------------------------
    print_header("BAGIAN 4: HYPERPARAMETER TUNING & FEATURE SELECTION")
    hasil_tuning, best_model = run_tuning(
        X_train, X_test, y_train, y_test,
        rf_before=rf_model,
        hasil_prep=hasil_prep
    )

    # --------------------------------------------------
    # RINGKASAN AKHIR
    # --------------------------------------------------
    print_header("RINGKASAN AKHIR")
    import pandas as pd
    hasil_direct.index = [f"{i} (Direct)" for i in hasil_direct.index]
    df_semua = pd.concat([hasil_direct, hasil_prep, hasil_tuning])
    print(df_semua.round(4).to_string())

    best_idx = df_semua['F1 Score'].idxmax()
    best_row = df_semua.loc[[best_idx]].iloc[0]
    print(f"\n  Model Terbaik   : {best_idx}")
    print(f"  F1 Score        : {best_row['F1 Score']:.4f}")
    print(f"  Akurasi         : {best_row['Akurasi']:.4f}")
    print(f"\n  Model disimpan di: output/models/best_model.pkl")
    print(f"  Plot  disimpan di: output/plots/")
    print("\n[SELESAI] Semua tahap berhasil dijalankan!")


if __name__ == "__main__":
    main()
