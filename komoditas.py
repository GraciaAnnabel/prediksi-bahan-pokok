import streamlit as st  # Harus paling atas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Konfigurasi halaman
st.set_page_config(layout="wide")
st.title("Prediksi Harga Komoditas Bahan Pokok")

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel harga komoditas", type=["xlsx"])

if uploaded_file:
    # Load dan siapkan data
    df_raw = pd.read_excel(uploaded_file)
    df_raw.set_index("BAHAN POKOK", inplace=True)
    df = df_raw.transpose()
    df.index = pd.to_datetime(df.index, format='%Y %B')

    st.subheader("Data Harga Komoditas")
    st.dataframe(df.tail(), use_container_width=True)

    # Normalisasi
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    # Fungsi buat sequence
    def create_sequences(data, n_input):
        X, y = [], []
        for i in range(len(data) - n_input):
            X.append(data[i:i+n_input])
            y.append(data[i+n_input])
        return np.array(X), np.array(y)

    # Parameter prediksi
    n_input = 5
    n_forecast = 6
    X, y = create_sequences(scaled_data, n_input)

    # Buat model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_input, df.shape[1])))
    model.add(Dense(df.shape[1]))
    model.compile(optimizer='adam', loss='mse')

    with st.spinner("Melatih model..."):
        model.fit(X, y, epochs=300, verbose=0)

    # Prediksi 6 bulan ke depan (recursive)
    last_seq = scaled_data[-n_input:]
    preds_scaled = []

    for _ in range(n_forecast):
        input_seq = last_seq.reshape(1, n_input, df.shape[1])
        pred_scaled = model.predict(input_seq, verbose=0)[0]
        preds_scaled.append(pred_scaled)
        last_seq = np.vstack([last_seq[1:], pred_scaled])

    # Transform hasil prediksi
    preds = scaler.inverse_transform(preds_scaled)
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq='MS')
    df_preds = pd.DataFrame(np.round(preds).astype(int), columns=df.columns, index=future_dates)

    # === Pilih Komoditas ===
    st.subheader("Pilih Komoditas")
    selected_commodity = st.selectbox("Pilih salah satu komoditas", df.columns)

    st.markdown(f"### Prediksi Harga 6 Bulan ke Depan – **{selected_commodity}**")

    st.dataframe(df_preds[[selected_commodity]], use_container_width=True)

    # === Matplotlib Visualisasi ===
    st.subheader("Visualisasi Harga (Matplotlib)")

    full_df = pd.concat([df, df_preds])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(full_df[selected_commodity], marker='o', color='blue', label='Harga')
    ax.axvline(x=df.index[-1], color='red', linestyle='--', label='Mulai Prediksi')
    ax.set_ylabel("Harga (Rp)")
    ax.set_title(f"Tren Harga {selected_commodity}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # === Plotly Visualisasi Interaktif ===
    st.subheader(f"Tren Harga Prediksi – {selected_commodity} (Interaktif)")

    plot_df = df_preds[[selected_commodity]].copy()
    plot_df = plot_df.reset_index()
    plot_df.columns = ["Bulan", "Harga"]

    fig_plotly = px.line(
        plot_df,
        x="Bulan",
        y="Harga",
        title=f"Tren Harga Prediksi – {selected_commodity}",
        markers=True
    )

    fig_plotly.update_traces(line=dict(color="#4ac8ff", width=3))
    fig_plotly.update_layout(
        xaxis_title="Bulan",
        yaxis_title="Nilai (Rp)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="black", font_size=13),
        yaxis=dict(tickformat=",")
    )

    st.plotly_chart(fig_plotly, use_container_width=True)

else:
    st.info("Silakan unggah file Excel terlebih dahulu. Format: kolom = bulan, baris = komoditas.")
