import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras # Gunakan keras dari tensorflow agar sinkron
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# =========================================================
# 1. CONFIG & LOAD RESOURCES
# =========================================================
st.set_page_config(layout="wide", page_title="Dashboard TA - Forecasting WTI")

# PINDAHKAN KE SINI: Jalankan konfigurasi di tingkat global/paling atas
try:
    tf.keras.config.enable_unsafe_deserialization()
except AttributeError:
    # Untuk versi TF yang lebih lama jika method di atas tidak ada
    pass

@st.cache_resource
def load_resources():
    """Load model, scalers, dan parameter konfigurasi."""
    try:
        # Load Window Size
        window_size = 30 
        if os.path.exists('best_params_case_4.json'):
            with open('best_params_case_4.json', 'r') as f:
                params = json.load(f)
            window_size = params.get('sliding_window', 30)
            
        # Load Model dengan parameter tambahan
        if os.path.exists('best_model_case_4.h5'):
            # TAMBAHKAN safe_mode=False di sini
            model = tf.keras.models.load_model(
                "best_model_case_4.h5", 
                compile=False, 
                safe_mode=False
            )
        else:
            return None, None, None, None

model, scaler_X, scaler_y, window_size = load_resources()

# =========================================================
# 2. FUNGSI LOGIC (CORE ALGORITHM)
# =========================================================

def get_prediction_value(model, input_tensor):
    """Helper untuk mengambil nilai float tunggal dari prediksi model."""
    raw_output = model.predict(input_tensor, verbose=0)
    if isinstance(raw_output, list):
        pred_output = raw_output[0]
    else:
        pred_output = raw_output
    val_arr = np.array(pred_output)
    return float(val_arr.flatten()[0])

def predict_test_onestep(model, df_features, scaler_X, window_size, test_size):
    """Melakukan prediksi pada data testing dengan sliding window."""
    total_needed = test_size + window_size
    if len(df_features) < total_needed:
        return None

    data_for_testing = df_features.iloc[-total_needed:].values
    data_scaled = scaler_X.transform(data_for_testing)
    
    X_test = []
    for i in range(test_size):
        window = data_scaled[i : i + window_size]
        X_test.append(window)
        
    X_test = np.array(X_test)
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    if isinstance(y_pred_scaled, list):
        y_pred_scaled = y_pred_scaled[0]
        
    return np.array(y_pred_scaled)

def forecast_future_recursive(model, initial_sequence_scaled, horizon, window_size):
    """Forecasting masa depan secara rekursif."""
    current_sequence = initial_sequence_scaled.copy()
    pred_results = []
    
    last_gprd_val = current_sequence[-1, 1] 

    for _ in range(horizon):
        input_tensor = current_sequence.reshape(1, window_size, 2)
        pred_val = get_prediction_value(model, input_tensor)
        pred_results.append(pred_val)
        
        new_step = np.array([[pred_val, last_gprd_val]])
        current_sequence = np.vstack([current_sequence[1:], new_step])
        
    return np.array(pred_results)

# =========================================================
# 3. UI & NAVIGASI
# =========================================================

if 'data' not in st.session_state:
    st.session_state['data'] = None

st.sidebar.title("Menu Dashboard")
menu = st.sidebar.selectbox(
    "Navigasi:",
    ["Homepage", "Import Data", "Statistika Deskriptif", "Forecasting WTI"]
)

if model is not None:
    st.sidebar.success(f"Model Ready (Window: {window_size})")
else:
    st.sidebar.warning("Model files not found.")

# =========================================================
# 4. KONTEN HALAMAN
# =========================================================

if menu == "Homepage":
    st.title("Selamat Datang di Rancang Bangun Dashboard Peramalan Harga Minyak Mentah WTI")
    st.markdown("""
    Ini adalah halaman utama aplikasi dashboard aplikasi sebagai output akhir dari Proyek Analitik dari Ananda Moreno Reyhan Afriansyah.

    Aplikasi Dashboard ini bertujuan memberikan aplikasi yang sangat mudah digunakan user untuk melakukan peramalan harga minyak mentah WTI dengan model LSTM dan Attention Mechanism.

    ### Alur Penggunaan Aplikasi:

    1. **Import Data**: Unggah data yang berisikan harga minyak mentah WTI dalam USD (format bisa .csv atau .xlsx) melalui menu "Import Data".
       Pastikan data Anda memiliki kolom `Date` dan kolom target 'WTI'.

    2. **Statistika Deskriptif**: Setelah data diunggah, pilih menu Statistika Deskriptif. Menu ini Anda bisa melihat ringkasan statistik deskriptif, informasi tipe data dan line chart harga minyak mentah WTI.

    3. **Forecast**: Setelah melihat menu statistika deskriptif beralih pilih menu Forecast. Anda dapat memilih timestap horizon *forecast* yang anda inginkan dan aplikasi akan menampilkan perbandingan hasil prediksi model pada data uji serta hasil *forecast* untuk hari-hari berikutnya. Hasil forecast dapat di download berupa file excel.

    Silakan mulai perjalanan anda menggunakan Rancang Bangun Dashboard Peramalan Harga Minyak Mentah WTI semoga menyenangkan.

    ***'Moreno Reyhan Afriansyah'***
    """)
    st.info("Status: Siap digunakan")

elif menu == "Import Data":
    st.title("Import Dataset")
    uploaded_file = st.file_uploader("Upload File (CSV/Excel)", type=["xlsx", "csv"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            req_cols = ["WTI", "WTI lag 1", "GPRD lag 1"]
            if all(col in df.columns for col in req_cols):
                df = df.sort_index(ascending=True)
                st.session_state['data'] = df
                st.success(f"Data Berhasil Dimuat: {len(df)} baris.")
                st.dataframe(df.head())
            else:
                st.error(f"Format Kolom Salah. Wajib ada: {req_cols}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif menu == "Statistika Deskriptif":
    st.title("Eksplorasi Data")
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        col = st.selectbox("Pilih Variabel:", df.select_dtypes(include=np.number).columns)
        
        st.subheader(f"Tren {col}")
        st.line_chart(df[col])
        
        st.subheader("Statistik Ringkasan")
        st.dataframe(df.describe().T.style.format("{:.2f}"))
    else:
        st.warning("Silakan upload data terlebih dahulu di menu Import Data.")

elif menu == "Forecasting WTI":
    st.title("Forecasting & Evaluasi Model")
    
    if st.session_state['data'] is None or model is None:
        st.error("Data belum diimport atau Model tidak ditemukan.")
    else:
        data = st.session_state['data']
        total_rows = len(data)
        
        # --- INPUT PARAMETER ---
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                horizon = st.slider("Horizon Forecast (Hari ke depan):", 1, 30, 7)
            with c2:
                test_size = int(total_rows * 0.10)
                st.info(f"Menggunakan {test_size} hari terakhir sebagai Data Uji (Validasi).")

        if st.button("Jalankan Prediksi", type="primary"):
            with st.spinner("Sedang melakukan forecasting..."):
                try:
                    # 1. PERSIAPAN FITUR
                    fixed_features = ["WTI lag 1", "GPRD lag 1"]
                    df_features = data[fixed_features].copy()

                    # 2. PREDIKSI DATA TEST
                    pred_test_scaled = predict_test_onestep(
                        model, df_features, scaler_X, window_size, test_size
                    )
                    
                    if pred_test_scaled is None:
                        st.error("Data tidak cukup untuk melakukan prediksi dengan window size saat ini.")
                    else:
                        pred_test_real = scaler_y.inverse_transform(pred_test_scaled.reshape(-1, 1))
                        
                        actual_test = data['WTI'].iloc[-test_size:].values
                        dates_test = data.index[-test_size:]
                        
                        mape = mean_absolute_percentage_error(actual_test, pred_test_real)
                        rmse = np.sqrt(mean_squared_error(actual_test, pred_test_real))

                        # 3. FORECAST MASA DEPAN
                        last_seq_vals = df_features.iloc[-window_size:].values
                        last_seq_scaled = scaler_X.transform(last_seq_vals)
                        
                        pred_fut_scaled = forecast_future_recursive(model, last_seq_scaled, horizon, window_size)
                        pred_fut_real = scaler_y.inverse_transform(pred_fut_scaled.reshape(-1, 1))
                        dates_fut = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=horizon)

                        st.session_state['fcast_results'] = {
                            'mape': mape,
                            'rmse': rmse,
                            'dates_test': dates_test,
                            'actual_test': actual_test,
                            'pred_test': pred_test_real,
                            'dates_fut': dates_fut,
                            'pred_fut': pred_fut_real
                        }

                except Exception as e:
                    st.error(f"Runtime Error: {e}")

        # --- TAMPILKAN HASIL ---
        if 'fcast_results' in st.session_state:
            res = st.session_state['fcast_results']
            
            st.divider()
            st.subheader("Hasil Evaluasi")
            
            col1, col2 = st.columns(2)
            col1.metric("MAPE (Akurasi Error)", f"{res['mape']:.2%}")
            col2.metric("RMSE (Rata-rata Error USD)", f"${res['rmse']:.4f}")
            
            st.subheader("Visualisasi Perbandingan & Forecast")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(res['dates_test'], res['actual_test'], label="Data Aktual (Test Set)", color='black', alpha=0.6, linewidth=2)
            ax.plot(res['dates_test'], res['pred_test'], label="Prediksi Model (Validasi)", color='#1f77b4', linewidth=2)
            ax.plot(res['dates_fut'], res['pred_fut'], label=f"Forecast {len(res['dates_fut'])} Hari Kedepan", color='#d62728', linestyle='--', marker='o', markersize=4)
            
            ax.set_title("Perbandingan Aktual vs Prediksi & Forecast Masa Depan", fontsize=14)
            ax.set_ylabel("Harga WTI (USD)")
            ax.set_xlabel("Tanggal")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            
            df_future = pd.DataFrame({
                "Tanggal": res['dates_fut'],
                "Forecast_WTI": res['pred_fut'].flatten()
            })
            
            st.download_button(
                label="Download Hasil Forecast (CSV)",
                data=df_future.to_csv(index=False),
                file_name="forecast_wti_future.csv",
                mime="text/csv"

            )

