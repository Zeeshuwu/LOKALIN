import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import geocoder
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set the page title
st.set_page_config(page_title="LOKALIN - UMKM Dashboard", layout="wide")

# --- 1. Load Data UMKM ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Zeeshuwu/LOKALIN/main/UMKM%20Cilacap%20Benar.csv"
    df = pd.read_csv(url)
    if 'TK_TETAP' in df.columns and 'TK_LEPAS' in df.columns:
        df['PEKERJA'] = df['TK_TETAP'].fillna(0) + df['TK_LEPAS'].fillna(0)
    return df

df = load_data()

# --- 2. Load TFLite Model ---
@st.cache_resource
def load_model():
    model_path = "knn_umkm_model_fix.tflite"  # Local file
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()

# --- Sidebar Navigation ---
menu = st.sidebar.radio("Pilih Analisis:", [
    "🔍 UMKM Terdekat",
    "📊 Distribusi & Lapangan Kerja",
    "📈 Analisis Data UMKM",
    "🗂️ Semua Data UMKM"
])

# --- Menu 1: UMKM Terdekat ---
if menu == "🔍 UMKM Terdekat":
    st.header("🔍 Cari UMKM Terdekat dari Lokasi Kamu")
    g = geocoder.ip('me')
    if g.latlng:
        user_lat, user_lon = g.latlng
        st.success(f"Lokasi Kamu: Latitude {user_lat:.4f}, Longitude {user_lon:.4f}")
        user_location = np.array([[user_lon, user_lat]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], user_location)
        interpreter.invoke()
        indices = interpreter.get_tensor(output_details[0]['index'])
        st.subheader("🏠 5 UMKM Terdekat:")
        for idx in indices[:, 0]:
            umkm = df.iloc[idx]
            st.markdown(f"**{umkm['UMKM']}** - {umkm['USAHA']}")
            st.write(f"Lokasi: ({umkm['LAT']}, {umkm['LONG']})")
            st.markdown(f"[📍 Lihat di Google Maps](https://www.google.com/maps/search/?api=1&query={umkm['LAT']},{umkm['LONG']})")
            st.divider()
    else:
        st.error("Gagal mendeteksi lokasi.")

# --- Menu 2: Distribusi & Lapangan Kerja ---
elif menu == "📊 Distribusi & Lapangan Kerja":
    st.header("📊 Distribusi UMKM dan Penyerapan Tenaga Kerja")
    if 'USAHA' in df.columns and 'PEKERJA' in df.columns:
        st.subheader("Jumlah UMKM per Jenis Usaha")
        st.bar_chart(df['USAHA'].value_counts())

        st.subheader("Total Pekerja per Jenis Usaha")
        pekerja = df.groupby('USAHA')['PEKERJA'].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(6, len(pekerja) * 0.3)))
        sns.barplot(x=pekerja.values, y=pekerja.index, ax=ax)
        ax.set_xlabel("Jumlah Pekerja")
        ax.set_ylabel("Jenis Usaha")
        ax.set_title("Total Pekerja per Jenis Usaha")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Kolom 'USAHA', 'TK_TETAP', atau 'TK_LEPAS' tidak ditemukan.")

# --- Menu 3: Analisis Data UMKM ---
elif menu == "📈 Analisis Data UMKM":
    st.header("📈 Analisis Data UMKM Cilacap")
    st.subheader("Statistik Deskriptif")
    st.write(df.describe(include='all'))

    st.subheader("Peta Sebaran UMKM")
    if 'LAT' in df.columns and 'LONG' in df.columns:
        df_map = df[['LAT', 'LONG']].rename(columns={'LAT': 'latitude', 'LONG': 'longitude'})
        st.map(df_map)
    else:
        st.warning("Data tidak memuat informasi lokasi.")

    st.subheader("Filter Berdasarkan Jenis Usaha")
    usaha_list = df['USAHA'].dropna().unique().tolist()
    selected_usaha = st.multiselect("Pilih Jenis Usaha:", usaha_list)
    if selected_usaha:
        filtered = df[df['USAHA'].isin(selected_usaha)]
        st.write(f"Menampilkan {len(filtered)} UMKM.")
        st.dataframe(filtered)
    else:
        st.info("Silakan pilih jenis usaha.")

# --- Menu 4: Semua Data UMKM ---
elif menu == "🗂️ Semua Data UMKM":
    st.header("🗂️ Semua Data UMKM di Kabupaten Cilacap")
    st.subheader("🔎 Filter Data")
    kec_list = df['KEC.'].dropna().unique().tolist()
    usaha_list = df['USAHA'].dropna().unique().tolist()

    selected_kec = st.selectbox("Pilih Kecamatan:", ["Semua"] + kec_list)
    selected_usaha = st.selectbox("Pilih Jenis Usaha:", ["Semua"] + usaha_list)

    filtered_df = df.copy()
    if selected_kec != "Semua":
        filtered_df = filtered_df[filtered_df['KEC.'] == selected_kec]
    if selected_usaha != "Semua":
        filtered_df = filtered_df[filtered_df['USAHA'] == selected_usaha]

    st.write(f"📄 Menampilkan {len(filtered_df)} data UMKM")
    st.dataframe(filtered_df)

    st.subheader("⬇️ Unduh Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "umkm_cilacap.csv", "text/csv")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="UMKM")
    st.download_button("📥 Download Excel", excel_buffer.getvalue(), "umkm_cilacap.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
