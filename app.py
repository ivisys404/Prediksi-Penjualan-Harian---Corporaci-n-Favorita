import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import joblib
import os
from datetime import datetime

# Load model dan encoder dari file .pkl
model = joblib.load('model_gb.pkl')
le_family = joblib.load('le_family.pkl')
le_type = joblib.load('le_type.pkl')
le_city = joblib.load('le_city.pkl')
le_state = joblib.load('le_state.pkl')

st.set_page_config(page_title="Prediksi Penjualan", layout="wide")
st.title("Prediksi Penjualan Harian - Corporación Favorita")

# Sidebar input
st.sidebar.header("Input Data")

store_nbr = st.sidebar.number_input("Nomor Toko (1-54)", 1, 54, 1)
family = st.sidebar.selectbox("Kategori Produk", le_family.classes_)
onpromotion = st.sidebar.number_input("Jumlah produk promosi", 0, 1000, 0)
date = st.sidebar.date_input("Tanggal", datetime(2024,1,1))

# Data default toko (contoh, Anda bisa lengkapi sesuai stores.csv)
store_info = {
    1: {'type': 'D', 'city': 'Quito', 'state': 'Pichincha', 'cluster': 13},
}
info = store_info.get(store_nbr, {'type':'D','city':'Quito','state':'Pichincha','cluster':8})

type_store = st.sidebar.selectbox("Tipe Toko", le_type.classes_, index=list(le_type.classes_).index(info['type']))
city = st.sidebar.selectbox("Kota", le_city.classes_, index=list(le_city.classes_).index(info['city']))
state = st.sidebar.selectbox("Provinsi", le_state.classes_, index=list(le_state.classes_).index(info['state']))
cluster = st.sidebar.number_input("Cluster", value=info['cluster'])

# Fitur kalender dari tanggal
year = date.year
month = date.month
day = date.day
dayofweek = date.weekday()
dayofyear = date.timetuple().tm_yday
weekofyear = date.isocalendar()[1]
is_weekend = 1 if dayofweek >= 5 else 0

oil_price = st.sidebar.number_input("Harga Minyak (USD)", 0.0, 200.0, 50.0)
is_holiday = st.sidebar.checkbox("Hari Libur Nasional?", False)

# Lag features – untuk demo, isi default. Di produksi bisa dari database.
sales_lag7 = st.sidebar.number_input("Penjualan 7 hari lalu", 0.0, 10000.0, 0.0)
sales_lag14 = st.sidebar.number_input("Penjualan 14 hari lalu", 0.0, 10000.0, 0.0)
sales_roll7 = st.sidebar.number_input("Rata2 penjualan 7 hari", 0.0, 10000.0, 0.0)
sales_roll28 = st.sidebar.number_input("Rata2 penjualan 28 hari", 0.0, 10000.0, 0.0)
sales_std7 = st.sidebar.number_input("Std penjualan 7 hari", 0.0, 1000.0, 0.0)

# Encoding kategori
family_enc = le_family.transform([family])[0]
type_enc = le_type.transform([type_store])[0]
city_enc = le_city.transform([city])[0]
state_enc = le_state.transform([state])[0]

# Buat dataframe input
input_df = pd.DataFrame([[
    store_nbr, family_enc, type_enc, cluster,
    city_enc, state_enc,
    year, month, day, dayofweek, dayofyear, weekofyear,
    is_weekend, is_holiday, onpromotion,
    oil_price,
    sales_lag7, sales_lag14, sales_roll7, sales_roll28, sales_std7
]], columns=[
    'store_nbr','family_enc','type_enc','cluster',
    'city_enc','state_enc',
    'year','month','day','dayofweek','dayofyear','weekofyear',
    'is_weekend','is_holiday','onpromotion',
    'oil_price',
    'sales_lag7','sales_lag14','sales_roll7','sales_roll28','sales_std7'
])

# Tombol prediksi
if st.sidebar.button("Prediksi Penjualan"):
    pred_log = model.predict(input_df)[0]
    pred_sales = np.expm1(pred_log)
    st.success(f"Prediksi Penjualan: **${pred_sales:,.2f}**")

# URL download model
MODEL_URL = "https://drive.google.com/drive/folders/1264LL0qiKLpsLqUAud4WwtqfyN8TYGY5?usp=sharing"
MODEL_PATH = "model_gb.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model (sekitar 2-3 menit)..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return joblib.load(MODEL_PATH)

model = load_model()