import streamlit as st
import pandas as pd
import numpy as np
import joblib  # atau pickle, tergantung cara menyimpan model
import os

# Judul aplikasi
st.title("Prediksi Income Berdasarkan Umur dan Pengalaman Kerja")

# Deskripsi
st.markdown("""
Aplikasi ini memprediksi pendapatan (Income) berdasarkan:
- **Age (Usia)**
- **Experience (Pengalaman kerja)**
""")

# Load model
model_path = "model.pkl"  # Sesuaikan dengan nama file model kamu
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model belum tersedia. Pastikan file 'model.pkl' ada di direktori.")
    st.stop()

# Input dari pengguna
age = st.number_input("Masukkan usia (Age):", min_value=0.0, format="%.1f")
experience = st.number_input("Masukkan pengalaman kerja (Experience):", min_value=0.0, format="%.1f")

# Prediksi ketika tombol ditekan
if st.button("Prediksi Income"):
    try:
        new_data = pd.DataFrame([[age, experience]], columns=["Age", "Experience"])
        predicted_income = model.predict(new_data)

        st.success(f"Prediksi Income adalah: **${predicted_income[0][0]:,.2f}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
