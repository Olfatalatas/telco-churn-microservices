import streamlit as st
import requests

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn AI", page_icon="🔮", layout="wide")

st.title("📊 Telco Customer Churn Predictor")
st.write("Masukkan profil pelanggan di bawah ini. Sistem akan mengirim data ke server AI (FastAPI) untuk dianalisis secara real-time.")
st.markdown("---")

# Menggunakan kolom agar form tidak terlalu panjang ke bawah
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Data Demografi")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen (Lansia)?", ["Yes", "No"])
    Partner = st.selectbox("Punya Pasangan?", ["Yes", "No"])
    Dependents = st.selectbox("Punya Tanggungan (Anak)?", ["Yes", "No"])
    tenure = st.number_input("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=1)

with col2:
    st.header("📱 Layanan Utama")
    PhoneService = st.selectbox("Layanan Telepon?", ["Yes", "No"])
    MultipleLines = st.selectbox("Banyak Saluran?", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Tipe Internet", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Keamanan Online?", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Backup Online?", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Proteksi Perangkat?", ["Yes", "No", "No internet service"])

with col3:
    st.header("💳 Tagihan & Kontrak")
    TechSupport = st.selectbox("Dukungan Teknis?", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV?", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies?", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Tagihan Tanpa Kertas?", ["Yes", "No"])
    PaymentMethod = st.selectbox("Metode Pembayaran", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Tagihan Bulanan ($)", min_value=0.0, value=29.85)
    TotalCharges = st.number_input("Total Tagihan ($)", min_value=0.0, value=29.85)

st.markdown("---")

# TOMBOL PREDIKSI
if st.button("🚀 Prediksi Sekarang", use_container_width=True):

    SeniorCitizen_API = 1 if SeniorCitizen == "Yes" else 0

    # 1. Bungkus data dari form ke dalam format JSON (Dictionary Python)
    data_paket = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen_API,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # 2. Kurir (requests) mengirim paket ke alamat FastAPI
    # Pastikan Uvicorn / main_api.py sedang menyala di port 8000!
    URL_API = "http://localhost:8000/predict"
    
    with st.spinner("AI sedang berpikir..."):
        try:
            response = requests.post(URL_API, json=data_paket)
            
            # Jika pengiriman sukses (Status Code 200)
            if response.status_code == 200:
                hasil = response.json()
                
                # Tampilkan hasil ke layar Web
                st.subheader("Hasil Analisis AI:")
                if hasil['prediction_code'] == 1:
                    st.error(f"⚠️ **RISIKO TINGGI!** {hasil['message']}")
                    st.warning("Saran: Segera kirimkan promo diskon atau hubungi pelanggan ini.")
                else:
                    st.success(f"✅ **AMAN.** {hasil['message']}")
            else:
                st.error(f"Error dari API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 GAGAL TERHUBUNG KE API! Pastikan kamu sudah menjalankan 'uvicorn main_api:app' di terminal lain.")