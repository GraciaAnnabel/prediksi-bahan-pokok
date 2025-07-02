import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Konfigurasi Streamlit
st.set_page_config(layout="wide")
st.title("📈 Prediksi Harga Komoditas Bahan Pokok (LSTM)")

# Upload file
uploaded_file = st.file_uploader("📥 Unggah file Excel harga komoditas", type=["xlsx"])

if uploaded_file:
    # Load data
    df_raw = pd.read_excel(uploaded_file)
    df_raw.set_index("BAHAN POKOK", inplace=True)
    df = df_raw.transpose()

    # === Konversi nama bulan Indonesia ke Inggris ===
    bulan_mapping = {
        'JANUARI': 'January', 'FEBRUARI': 'February', 'MARET': 'March',
        'APRIL': 'April', 'MEI': 'May', 'JUNI': 'June', 'JULI': 'July',
        'AGUSTUS': 'August', 'SEPTEMBER': 'September', 'OKTOBER': 'October',
        'NOVEMBER': 'November', 'DESEMBER': 'December'
    }

    df.index = df.index.str.strip().str.upper()
    for indo, eng in bulan_mapping.items():
        df.index = df.index.str.replace(indo, eng, regex=False)

    # Ubah index ke datetime
    df.index = pd.to_datetime(df.index, format='%Y %B', errors='coerce')
    df = df[~df.index.isna()]  # Hapus baris dengan index yang gagal

    # Tampilkan data
    st.subheader("📊 Data Harga Komoditas")
    st.dataframe(df.tail(), use_container_width=True)

    # Normalisasi
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    # Buat input-output sequences
    def create_sequences(data, n_input):
        X, y = [], []
        for i in range(len(data) - n_input):
            X.append(data[i:i+n_input])
            y.append(data[i+n_input])
        return np.array(X), np.array(y)

    n_input = 5
    n_forecast = 6
    X, y = create_sequences(scaled_data, n_input)

    # Buat model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_input, df.shape[1])))
    model.add(Dense(df.shape[1]))
    model.compile(optimizer='adam', loss='mse')

    with st.spinner("🔁 Melatih model..."):
        model.fit(X, y, epochs=300, verbose=0)

    # Prediksi 6 bulan ke depan
    last_seq = scaled_data[-n_input:]
    preds_scaled = []

    for _ in range(n_forecast):
        input_seq = last_seq.reshape(1, n_input, df.shape[1])
        pred_scaled = model.predict(input_seq, verbose=0)[0]
        preds_scaled.append(pred_scaled)
        last_seq = np.vstack([last_seq[1:], pred_scaled])

    preds = scaler.inverse_transform(preds_scaled)
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq='MS')
    df_preds = pd.DataFrame(np.round(preds).astype(int), columns=df.columns, index=future_dates)

    # === Pilih Komoditas ===
    selected_commodity = st.selectbox("📌 Pilih Komoditas untuk Menampilkan Tabel Prediksi", df.columns)

    # === Tampilkan hanya tabel prediksi dari komoditas terpilih
    st.subheader(f"📅 Prediksi Harga 6 Bulan ke Depan – {selected_commodity}")
    st.dataframe(df_preds[[selected_commodity]], use_container_width=True)

    # Gabungkan dengan data lama untuk grafik
    full_df = pd.concat([df, df_preds])

    # Plot visualisasi prediksi
    st.subheader("📉 Visualisasi Harga")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(full_df[selected_commodity], marker='o', color='blue', label='Harga')
    ax.axvline(x=df.index[-1], color='red', linestyle='--', label='Awal Prediksi')
    ax.set_ylabel("Harga (Rp)")
    ax.set_title(f"Tren Harga {selected_commodity}")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"Rp {x:,.0f}".replace(",", ".")))
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

else:
    st.info("📂 Silakan unggah file Excel terlebih dahulu. Format: baris = komoditas, kolom = bulan seperti '2023 Januari'.")
