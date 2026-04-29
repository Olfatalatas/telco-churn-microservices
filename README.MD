# 🔮 Telco Customer Churn Prediction - Microservices Architecture

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

## 📖 Deskripsi Proyek
Proyek ini adalah implementasi *End-to-End Machine Learning* untuk memprediksi probabilitas *churn* (berhenti berlangganan) pelanggan pada perusahaan telekomunikasi. Dibangun dengan pendekatan **Microservices Architecture**, proyek ini memisahkan antara proses *training*, *backend API*, dan *frontend UI* untuk memastikan skalabilitas, fleksibilitas, dan kemudahan pemeliharaan sesuai standar praktik terbaik **MLOps**.

Sistem AI ini dilatih menggunakan algoritma **XGBoost** berkinerja tinggi dan memberikan hasil prediksi secara *real-time* melalui antarmuka web interaktif yang intuitif.

## 🏗️ Arsitektur Sistem
Sistem ini terdiri dari 4 komponen utama yang saling terintegrasi:
1. **EDA & Data Processing (`eda_churn.py`)**: Bertanggung jawab untuk eksplorasi data, penanganan anomali data, dan pemahaman karakteristik pelanggan sebelum pemodelan.
2. **Model Training Pipeline (`train_churn.py`)**: Pipeline *machine learning* untuk menyiapkan data, merekayasa fitur (*Label Encoding*), melatih model XGBoost, dan mengekspor model/aset ke dalam format biner yang efisien (`.pkl`).
3. **Backend API Server (`main_api.py`)**: *RESTful API* menggunakan **FastAPI** yang memuat model ke dalam *memory* (RAM) saat *startup*, dan melayani *request* prediksi dengan latensi sangat rendah. Dilengkapi dengan lapisan keamanan validasi *schema* input menggunakan **Pydantic**.
4. **Frontend Web UI (`app_frontend.py`)**: Antarmuka interaktif yang dibangun menggunakan **Streamlit** bagi tim bisnis/marketing untuk menginput profil pelanggan dan mendapatkan hasil analisis serta tingkat risiko *churn* secara instan.

## 🛠️ Teknologi Utama (Tech Stack)
* **Machine Learning & Data Science**: `scikit-learn`, `xgboost`, `pandas`, `joblib`
* **API Backend**: `FastAPI`, `Uvicorn`, `Pydantic`
* **Web Frontend**: `Streamlit`, `requests`

## 📁 Struktur Direktori
```text
churn_prediction/
├── data/                  # Folder dataset mentah & hasil pemrosesan
├── env/                   # Virtual environment Python (diabaikan dalam versi kontrol)
├── models/                # File model & encoder hasil training (churn_xgb_model.pkl, dll)
├── visualizations/        # Berisi grafik hasil proses EDA
├── app_frontend.py        # Streamlit App (User Interface)
├── eda_churn.py           # Skrip Exploratory Data Analysis
├── main_api.py            # FastAPI Application (API Backend)
├── train_churn.py         # Skrip pelatihan Machine Learning model
├── requirements.txt       # Daftar dependensi package Python
└── README.MD              # Dokumentasi repositori ini
```

## 🚀 Cara Instalasi dan Menjalankan Proyek di Lingkungan Lokal

### 1. Persiapan Environment
Pastikan Python (versi 3.8 ke atas) sudah terinstal di sistem Anda.

Kloning repositori ini ke dalam *machine* Anda:
```bash
git clone https://github.com/Olfatalatas/telco-churn-microservices.git
cd telco-churn-microservices/churn_prediction
```

Buat dan aktifkan *Virtual Environment* (Direkomendasikan):
```bash
python -m venv env

# Pengguna Windows:
env\Scripts\activate
# Pengguna Mac/Linux:
source env/bin/activate
```

Instal seluruh dependensi yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Backend API (FastAPI Server)
Buka terminal baru, pastikan *virtual environment* sudah aktif, kemudian jalankan server backend:
```bash
uvicorn main_api:app --reload
```
Server API akan berjalan secara lokal di `http://localhost:8000`. Anda dapat mengakses dokumentasi API interaktif secara langsung (*Swagger UI*) dengan mengunjungi `http://localhost:8000/docs`.

### 3. Menjalankan Frontend UI (Streamlit)
Buka terminal kedua, pastikan *virtual environment* aktif, kemudian jalankan aplikasi *frontend*:
```bash
streamlit run app_frontend.py
```
Aplikasi web klien akan otomatis terbuka di *browser* pada alamat `http://localhost:8501`.

### 4. Melatih Ulang Model AI (Opsional)
Jika terdapat dataset baru atau Anda ingin mencoba *hyperparameter* lain, Anda dapat melatih ulang model AI dengan menjalankan:
```bash
python train_churn.py
```
Skrip ini akan memperbarui *artifacts* di dalam folder `models/`. Setelah berhasil dieksekusi, pastikan Anda **me-restart** server FastAPI agar model terbaru dimuat ulang ke dalam sistem.

## 📡 Referensi API (*API Reference*)

### Endpoint Prediksi Churn Pelanggan
* **URL/Endpoint:** `/predict`
* **Method:** `POST`
* **Deskripsi:** Menerima payload berupa profil pelanggan (berformat JSON) dan mengembalikan hasil analisis sistem.

**Contoh *Request Payload*:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

**Contoh *Response* (Berhasil):**
```json
{
  "status": "success",
  "prediction_code": 1,
  "prediction_label": "Churn (Akan Kabur)",
  "churn_probability": 85.5,
  "message": "Sistem AI yakin 85.5% pelanggan ini akan kabur."
}
```

## 💼 Nilai Bisnis (*Business Use Case*)
Sistem prediksi *churn* mikroservis ini merupakan aset yang berharga bagi departemen **Customer Retention** atau **Marketing**. Dengan memasukkan profil pengguna ke sistem (secara manual atau melalui sistem otomatis/CRM via API), perusahaan dapat mengambil inisiatif proaktif seperti memberikan promo penawaran spesial, insentif, atau dukungan khusus sebelum pelanggan beralih ke kompetitor.

---
*Dikembangkan untuk mendemonstrasikan kapabilitas implementasi Data Science, Model Deployment, dan AI Engineering Praktis.*
