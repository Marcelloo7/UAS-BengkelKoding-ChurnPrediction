# ============================================================
# deployment/app.py
# Streamlit App - Customer Churn Prediction
# UAS Bengkel Koding Data Science
# v2 - Tambah preset profil & threshold adjustment
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'output', 'models')

@st.cache_resource
def load_artifacts():
    model   = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    le_dict = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    return model, scaler, le_dict

model, scaler, le_dict = load_artifacts()

# ============================================================
# PRESET PROFIL
# Diambil dari median dataset (EDA notebook)
# ============================================================
PRESETS = {
    "— Pilih Preset —": None,
    "⚠️ Profil Berpotensi Churn": {
        # Nilai dari median baris churn=1 di dataset
        "gender": "Female",
        "age": 35,
        "country": "India",
        "acquisition_channel": "Organic",
        "device_type": "Mobile",
        "subscription_type": "Monthly",
        "is_premium_user": 0,
        "payment_method": "BKash",
        "total_spent": 150.0,        # jauh di bawah median non-churn (524)
        "avg_order_value": 60.58,
        "discount_used": 0,
        "total_visits": 15,
        "avg_session_time": 8.21,
        "pages_per_session": 4.0,
        "email_open_rate": 0.49,
        "email_click_rate": 0.25,
        "satisfaction_score": 1.0,   # sangat rendah
        "nps_score": 5,
        "support_tickets": 6,        # tinggi
        "refund_requested": 1,       # pernah refund
        "delivery_delay_days": 5,    # delay tinggi
        "marketing_spend_per_user": 17.40,
        "lifetime_value": 1225.0,
        "last_3_month_purchase_freq": 2,  # jarang beli
    },
    "✅ Profil Pelanggan Loyal": {
        # Nilai dari median baris churn=0 di dataset
        "gender": "Male",
        "age": 35,
        "country": "Germany",
        "acquisition_channel": "Organic",
        "device_type": "Tablet",
        "subscription_type": "Annual",
        "is_premium_user": 1,
        "payment_method": "UPI",
        "total_spent": 524.77,
        "avg_order_value": 59.97,
        "discount_used": 1,
        "total_visits": 15,
        "avg_session_time": 7.97,
        "pages_per_session": 4.0,
        "email_open_rate": 0.50,
        "email_click_rate": 0.25,
        "satisfaction_score": 5.0,   # tinggi
        "nps_score": 5,
        "support_tickets": 1,        # jarang komplain
        "refund_requested": 0,
        "delivery_delay_days": 2,
        "marketing_spend_per_user": 17.66,
        "lifetime_value": 1214.54,
        "last_3_month_purchase_freq": 10,  # sering beli
    },
    "📊 Nilai Median Dataset": {
        # Nilai median keseluruhan
        "gender": "Male",
        "age": 35,
        "country": "Germany",
        "acquisition_channel": "Organic",
        "device_type": "Tablet",
        "subscription_type": "Monthly",
        "is_premium_user": 0,
        "payment_method": "UPI",
        "total_spent": 498.84,
        "avg_order_value": 60.11,
        "discount_used": 0,
        "total_visits": 15,
        "avg_session_time": 7.99,
        "pages_per_session": 4.0,
        "email_open_rate": 0.50,
        "email_click_rate": 0.25,
        "satisfaction_score": 4.0,
        "nps_score": 5,
        "support_tickets": 2,
        "refund_requested": 0,
        "delivery_delay_days": 3,
        "marketing_spend_per_user": 17.63,
        "lifetime_value": 1216.21,
        "last_3_month_purchase_freq": 7,
    },
}

# ============================================================
# HEADER
# ============================================================
st.title("📊 Customer Churn Prediction")
st.markdown("**UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro**")
st.markdown("Prediksi apakah pelanggan akan **churn** (berhenti) atau tetap menggunakan layanan.")
st.divider()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.success("✅ Model berhasil dimuat!")
    st.markdown("""
    **Model:** Random Forest (Tuned)
    **Preprocessing:** SMOTE + StandardScaler
    **F1 Score:** ~0.60 | **Akurasi:** ~85%
    """)
    st.divider()

    st.markdown("""
    **🔑 Top 3 Fitur Paling Berpengaruh:**
    1. 🥇 Satisfaction Score
    2. 🥈 Total Spent
    3. 🥉 Support Tickets
    """)
    st.divider()

    st.subheader("⚙️ Pengaturan Threshold")
    threshold = st.slider(
        "Threshold Prediksi Churn",
        min_value=0.10, max_value=0.60,
        value=0.30, step=0.05,
        help="Default 0.5 terlalu tinggi karena data asli hanya 15% churn (imbalance). "
             "Threshold 0.30 lebih sensitif mendeteksi potensi churn."
    )
    st.caption(f"Prediksi CHURN jika probabilitas ≥ **{threshold:.0%}**")
    st.info("💡 Turunkan threshold → lebih banyak terdeteksi churn (lebih sensitif).\n\n"
            "Naikkan threshold → lebih ketat, hanya yang benar-benar berisiko tinggi.")
    st.divider()
    st.caption("Dibuat oleh: Marcello | A11.2023.15390")

# ============================================================
# PRESET SELECTOR
# ============================================================
st.subheader("🚀 Quick Load: Preset Profil Pelanggan")
col_preset1, col_preset2 = st.columns([2, 3])

with col_preset1:
    preset_choice = st.selectbox(
        "Pilih preset untuk mengisi form otomatis:",
        options=list(PRESETS.keys()),
        key="preset_selector"
    )

with col_preset2:
    if preset_choice != "— Pilih Preset —":
        st.markdown(" ")
        if preset_choice == "⚠️ Profil Berpotensi Churn":
            st.warning("Profil ini: satisfaction rendah, support tickets tinggi, total spent rendah → kemungkinan besar CHURN")
        elif preset_choice == "✅ Profil Pelanggan Loyal":
            st.success("Profil ini: satisfaction tinggi, premium user, annual subscription → kemungkinan besar TIDAK CHURN")
        else:
            st.info("Nilai tengah dari seluruh dataset — hasil prediksi bisa ke salah satu arah.")

# Ambil nilai preset yang dipilih
_selected = PRESETS.get(preset_choice)
P = _selected if (_selected is not None) else PRESETS["📊 Nilai Median Dataset"]

st.divider()
st.subheader("📝 Data Pelanggan")

# ============================================================
# FORM INPUT — 3 KOLOM
# ============================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Profil Pelanggan**")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                          index=["Male","Female","Other"].index(P["gender"]))
    age    = st.number_input("Usia", min_value=0, max_value=100, value=int(P["age"]))
    country = st.selectbox("Negara",
                           ["USA", "UK", "India", "Germany", "Bangladesh"],
                           index=["USA","UK","India","Germany","Bangladesh"].index(P["country"]))
    acquisition_channel = st.selectbox(
        "Sumber Pelanggan",
        ["Organic", "Email", "Google Ads", "Facebook Ads", "Referral"],
        index=["Organic","Email","Google Ads","Facebook Ads","Referral"].index(P["acquisition_channel"])
    )
    device_type = st.selectbox(
        "Jenis Perangkat", ["Mobile", "Desktop", "Tablet"],
        index=["Mobile","Desktop","Tablet"].index(P["device_type"])
    )

with col2:
    st.markdown("**💳 Langganan & Transaksi**")
    subscription_type = st.selectbox(
        "Tipe Langganan", ["Monthly", "Annual"],
        index=["Monthly","Annual"].index(P["subscription_type"])
    )
    is_premium_user = st.selectbox(
        "Pengguna Premium?", [0, 1],
        index=P["is_premium_user"],
        format_func=lambda x: "Ya ✨" if x == 1 else "Tidak"
    )
    payment_method = st.selectbox(
        "Metode Pembayaran",
        ["Card", "PayPal", "UPI", "BKash", "SEPA"],
        index=["Card","PayPal","UPI","BKash","SEPA"].index(P["payment_method"])
    )
    total_spent = st.number_input(
        "Total Pengeluaran ($)", min_value=0.0, value=float(P["total_spent"]), step=10.0
    )
    avg_order_value = st.number_input(
        "Rata-rata Nilai Transaksi", min_value=0.0, value=float(P["avg_order_value"]), step=5.0
    )
    discount_used = st.selectbox(
        "Menggunakan Diskon?", [0, 1],
        index=P["discount_used"],
        format_func=lambda x: "Ya" if x == 1 else "Tidak"
    )

with col3:
    st.markdown("**📈 Aktivitas & Kepuasan**")
    total_visits      = st.number_input("Total Kunjungan", min_value=0, value=int(P["total_visits"]))
    avg_session_time  = st.number_input("Rata-rata Waktu Sesi (menit)", min_value=0.0,
                                        value=float(P["avg_session_time"]))
    pages_per_session = st.number_input("Halaman per Sesi", min_value=0.0,
                                        value=float(P["pages_per_session"]))
    email_open_rate   = st.slider("Email Open Rate", 0.0, 1.0, float(P["email_open_rate"]), step=0.01,
                                   help="Skala 0.0 – 1.0 (contoh: 0.50 = 50% email dibuka)")
    email_click_rate  = st.slider("Email Click Rate", 0.0, 1.0, float(P["email_click_rate"]), step=0.01,
                                   help="Skala 0.0 – 1.0 (contoh: 0.25 = 25% klik)")

    # 3 FITUR PALING BERPENGARUH — diberi highlight
    st.markdown("---")
    st.markdown("**🔑 Fitur Kunci:**")
    satisfaction_score = st.slider(
        "⭐ Skor Kepuasan (1–5)", 1.0, 5.0, float(P["satisfaction_score"]), step=0.5,
        help="Fitur #1 paling berpengaruh. Skor rendah → risiko churn tinggi."
    )
    nps_score = st.slider("NPS Score", -100, 100, int(P["nps_score"]))

st.divider()
col4, col5 = st.columns(2)

with col4:
    st.markdown("**🚚 Layanan & Dukungan**")
    support_tickets = st.number_input(
        "🎫 Jumlah Tiket Support", min_value=0, value=int(P["support_tickets"]),
        help="Fitur #3 paling berpengaruh. Makin banyak tiket → risiko churn lebih tinggi."
    )
    refund_requested    = st.selectbox("Pernah Minta Refund?", [0, 1],
                                       index=P["refund_requested"],
                                       format_func=lambda x: "Ya" if x==1 else "Tidak")
    delivery_delay_days = st.number_input("Keterlambatan Pengiriman (hari)", min_value=0,
                                          value=int(P["delivery_delay_days"]))

with col5:
    st.markdown("**💰 Nilai Pelanggan**")
    marketing_spend_per_user   = st.number_input("Biaya Marketing per User", min_value=0.0,
                                                  value=float(P["marketing_spend_per_user"]))
    lifetime_value             = st.number_input("Lifetime Value", min_value=0.0,
                                                  value=float(P["lifetime_value"]))
    last_3_month_purchase_freq = st.number_input(
        "Frekuensi Beli 3 Bulan Terakhir", min_value=0,
        value=int(P["last_3_month_purchase_freq"]),
        help="Makin jarang beli → sinyal churn."
    )

# ============================================================
# TOMBOL PREDIKSI
# ============================================================
st.divider()
predict_btn = st.button("🔍 Prediksi Churn", type="primary", use_container_width=True)

if predict_btn:
    input_dict = {
        'gender'                    : gender,
        'age'                       : age,
        'country'                   : country,
        'acquisition_channel'       : acquisition_channel,
        'device_type'               : device_type,
        'subscription_type'         : subscription_type,
        'is_premium_user'           : is_premium_user,
        'total_visits'              : total_visits,
        'avg_session_time'          : avg_session_time,
        'pages_per_session'         : pages_per_session,
        'email_open_rate'           : email_open_rate,
        'email_click_rate'          : email_click_rate,
        'total_spent'               : total_spent,
        'avg_order_value'           : avg_order_value,
        'discount_used'             : discount_used,
        'support_tickets'           : support_tickets,
        'refund_requested'          : refund_requested,
        'delivery_delay_days'       : delivery_delay_days,
        'payment_method'            : payment_method,
        'satisfaction_score'        : satisfaction_score,
        'nps_score'                 : nps_score,
        'marketing_spend_per_user'  : marketing_spend_per_user,
        'lifetime_value'            : lifetime_value,
        'last_3_month_purchase_freq': last_3_month_purchase_freq,
    }
    input_df = pd.DataFrame([input_dict])

    # Encoding kolom kategorikal
    cat_cols = ['gender', 'country', 'acquisition_channel',
                'device_type', 'subscription_type', 'payment_method']
    for col in cat_cols:
        if col in le_dict:
            le  = le_dict[col]
            val = input_df[col].values[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0

    # Scaling kolom numerik
    cols_scale = [
        'age', 'total_visits', 'avg_session_time', 'pages_per_session',
        'email_open_rate', 'email_click_rate', 'total_spent',
        'avg_order_value', 'support_tickets', 'delivery_delay_days',
        'satisfaction_score', 'nps_score', 'marketing_spend_per_user',
        'lifetime_value', 'last_3_month_purchase_freq'
    ]
    input_df[cols_scale] = scaler.transform(input_df[cols_scale])

    # Prediksi dengan threshold custom
    proba      = model.predict_proba(input_df)[0]
    prediction = 1 if proba[1] >= threshold else 0

    # ============================================================
    # TAMPILKAN HASIL
    # ============================================================
    st.subheader("📊 Hasil Prediksi")
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        if prediction == 1:
            st.error("## ⚠️ CHURN")
            st.markdown("Pelanggan ini **berpotensi churn.**")
        else:
            st.success("## ✅ TIDAK CHURN")
            st.markdown("Pelanggan ini kemungkinan **tetap loyal.**")

    with res_col2:
        st.metric("Probabilitas Churn", f"{proba[1]*100:.1f}%")
        st.caption(f"Threshold yang digunakan: {threshold:.0%}")

    with res_col3:
        st.metric("Probabilitas Tidak Churn", f"{proba[0]*100:.1f}%")

    with res_col4:
        risk_pct = proba[1] * 100
        if risk_pct >= 60:
            st.error("🔴 Risiko Tinggi")
        elif risk_pct >= threshold * 100:
            st.warning("🟠 Risiko Sedang")
        else:
            st.success("🟢 Risiko Rendah")

    # Progress bar probabilitas churn
    st.markdown(f"**Probabilitas Churn: {proba[1]*100:.1f}%**")
    st.progress(float(proba[1]))

    st.divider()

    # ============================================================
    # FAKTOR PENDORONG (berdasarkan input vs profil normal)
    # ============================================================
    st.subheader("🔍 Analisis Faktor Risiko")
    warnings = []
    positives = []

    if satisfaction_score <= 2.0:
        warnings.append(f"⚠️ Satisfaction Score sangat rendah ({satisfaction_score}/5) — median normal: 4.0")
    elif satisfaction_score <= 3.0:
        warnings.append(f"⚠️ Satisfaction Score di bawah rata-rata ({satisfaction_score}/5)")
    else:
        positives.append(f"✅ Satisfaction Score baik ({satisfaction_score}/5)")

    if total_spent < 200:
        warnings.append(f"⚠️ Total Spent sangat rendah (${total_spent:.0f}) — median non-churn: $524")
    elif total_spent < 350:
        warnings.append(f"⚠️ Total Spent di bawah rata-rata (${total_spent:.0f})")
    else:
        positives.append(f"✅ Total Spent normal (${total_spent:.0f})")

    if support_tickets >= 5:
        warnings.append(f"⚠️ Support Tickets tinggi ({support_tickets}) — menandakan banyak masalah")
    elif support_tickets >= 4:
        warnings.append(f"⚠️ Support Tickets di atas rata-rata ({support_tickets})")
    else:
        positives.append(f"✅ Support Tickets normal ({support_tickets})")

    if refund_requested == 1:
        warnings.append("⚠️ Pernah mengajukan refund — sinyal ketidakpuasan")

    if last_3_month_purchase_freq <= 2:
        warnings.append(f"⚠️ Frekuensi beli 3 bulan terakhir sangat rendah ({last_3_month_purchase_freq}x)")
    elif last_3_month_purchase_freq >= 8:
        positives.append(f"✅ Sering beli belakangan ini ({last_3_month_purchase_freq}x dalam 3 bulan)")

    if is_premium_user == 1:
        positives.append("✅ Pengguna Premium — lebih terikat dengan layanan")

    if subscription_type == "Annual":
        positives.append("✅ Langganan Tahunan — komitmen lebih kuat")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        if warnings:
            st.markdown("**🚩 Faktor Risiko:**")
            for w in warnings:
                st.markdown(w)
        else:
            st.markdown("**Tidak ada faktor risiko signifikan.**")

    with fcol2:
        if positives:
            st.markdown("**💚 Faktor Positif:**")
            for p in positives:
                st.markdown(p)

    st.divider()

    # ============================================================
    # REKOMENDASI
    # ============================================================
    st.subheader("💡 Rekomendasi Tindakan")
    if prediction == 1:
        st.warning("""
        **Pelanggan ini berisiko churn! Tindakan yang disarankan:**
        - 🎁 Berikan promo atau diskon retensi khusus
        - 📞 Hubungi pelanggan untuk mengetahui keluhan (terutama jika support tickets tinggi)
        - ⭐ Evaluasi pengalaman layanan — tingkatkan satisfaction score
        - 🔄 Tawarkan upgrade ke paket Annual dengan benefit lebih
        - 🔍 Investigasi alasan refund jika ada
        """)
    else:
        st.info("""
        **Pelanggan ini loyal! Pertahankan dengan:**
        - 🏆 Program loyalty reward untuk pelanggan setia
        - 📧 Kirim konten personal dan relevan
        - 🎯 Tawarkan fitur premium atau upsell yang sesuai profil
        - 📊 Monitor terus satisfaction score secara berkala
        """)