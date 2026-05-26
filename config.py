# ============================================================
# config.py
# Konfigurasi global project UAS Churn Prediction
# ============================================================

import os

# Path
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Sales - Marketing customer dataset.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
PLOTS_DIR   = os.path.join(OUTPUT_DIR, 'plots')
MODELS_DIR  = os.path.join(OUTPUT_DIR, 'models')

# Kolom
TARGET_COL  = 'churn'

COLS_DROP = [
    'customer_id', 'signup_date', 'last_purchase_date',
    'coupon_code', 'city'
]

COLS_KATEGORIKAL = [
    'gender', 'country', 'acquisition_channel',
    'device_type', 'subscription_type', 'payment_method'
]

COLS_BINER = ['is_premium_user', 'discount_used', 'refund_requested']

COLS_NUMERIK = [
    'age', 'total_visits', 'avg_session_time', 'pages_per_session',
    'email_open_rate', 'email_click_rate', 'total_spent',
    'avg_order_value', 'support_tickets', 'delivery_delay_days',
    'satisfaction_score', 'nps_score', 'marketing_spend_per_user',
    'lifetime_value', 'last_3_month_purchase_freq'
]

COLS_SCALE = [
    'age', 'total_visits', 'avg_session_time', 'pages_per_session',
    'email_open_rate', 'email_click_rate', 'total_spent',
    'avg_order_value', 'support_tickets', 'delivery_delay_days',
    'satisfaction_score', 'nps_score', 'marketing_spend_per_user',
    'lifetime_value', 'last_3_month_purchase_freq'
]

# Split
TEST_SIZE    = 0.2
RANDOM_STATE = 42
