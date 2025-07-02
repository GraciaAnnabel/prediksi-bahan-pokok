import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Konfigurasi Streamlit
st.set_page_config(layout="wide")
st.title("\ud83d\udcc8 Prediksi Harga Komoditas Bahan Pokok (LSTM - PyTorch)")

# Upload file
uploaded_file = st.file_uploader("\ud83d\udcc5 Unggah file Excel harga komoditas", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    df_raw.set_index("BAHAN POKOK", inplace=True)
    df = df_raw.transpose()

    bulan_mapping = {
        'JANUARI': 'January', 'FEBRUARI': 'February', 'MARET': 'March',
        'APRIL': 'April', 'MEI': 'May', 'JUNI': 'June', 'JULI': 'July',
        'AGUSTUS': 'August', 'SEPTEMBER': 'September', 'OKTOBER': 'October',
        'NOVEMBER': 'November', 'DESEMBER': 'December'
    }

    df.index = df.index.str.strip().str.upper()
    for indo, eng in bulan_mapping.items():
        df.index = df.index.str.replace(indo, eng, regex=False)

    df.index = pd.to_datetime(df.index, format='%Y %B', errors='coerce')
    df = df[~df.index.isna()]

    st.subheader("\ud83d\udcca Data Harga Komoditas")
    st.dataframe(df.tail(), use_container_width=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    def create_sequences(data, n_input):
        X, y = [], []
        for i in range(len(data) - n_input):
            X.append(data[i:i+n_input])
            y.append(data[i+n_input])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    n_input = 5
    n_forecast = 6
    X, y = create_sequences(scaled_data, n_input)

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, output_size=None):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size or input_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.linear(out)

    model = LSTMModel(input_size=df.shape[1], output_size=df.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with st.spinner("\ud83d\udd01 Melatih model..."):
        for epoch in range(300):
            model.train()
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    last_seq = torch.tensor(scaled_data[-n_input:], dtype=torch.float32).unsqueeze(0)
    preds_scaled = []

    for _ in range(n_forecast):
        with torch.no_grad():
            pred = model(last_seq)
        preds_scaled.append(pred.numpy().flatten())
        next_seq = torch.cat((last_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
        last_seq = next_seq

    preds = scaler.inverse_transform(np.array(preds_scaled))
    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq='MS')
    df_preds = pd.DataFrame(np.round(preds).astype(int), columns=df.columns, index=future_dates)

    selected_commodity = st.selectbox("\ud83d\udccc Pilih Komoditas untuk Menampilkan Tabel Prediksi", df.columns)

    st.subheader(f"\ud83d\udcc5 Prediksi Harga 6 Bulan ke Depan – {selected_commodity}")
    st.dataframe(df_preds[[selected_commodity]], use_container_width=True)

    full_df = pd.concat([df, df_preds])

    st.subheader("\ud83d\udcc9 Visualisasi Harga")
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
    st.info("\ud83d\udcc2 Silakan unggah file Excel terlebih dahulu. Format: baris = komoditas, kolom = bulan seperti '2023 Januari'.")
