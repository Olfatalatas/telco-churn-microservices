from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. INISIALISASI APLIKASI FASTAPI
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API MLOps untuk memprediksi apakah pelanggan akan kabur (Churn)",
    version="1.0.0"
)

# Variabel global untuk menyimpan "Otak" AI di RAM
model = None
encoders = None
features = None

# 2. LIFESPAN (KULKAS: Dijalankan SECARA OTOMATIS SAAT SERVER MENYALA)
@app.on_event("startup")
def load_assets():
    global model, encoders, features
    try:
        print("Membuka kulkas... me-load model dan aset ke memori (RAM)...")
        # Pastikan path ini sesuai dengan tempat kamu menyimpan .pkl di Tahap 1
        model = joblib.load('models/churn_xgb_model.pkl')
        encoders = joblib.load('models/churn_encoders.pkl')
        features = joblib.load('models/churn_features.pkl')
        print("✅ Semua aset berhasil di-load! Server siap menerima pesanan.")
    except Exception as e:
        print(f"❌ Gagal me-load aset. Pastikan kamu sudah menjalankan train_churn.py! Error: {e}")

# 3. PYDANTIC SCHEMA (SATPAM PINTU MASUK: Validasi Input User)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    # DUMMY DATA FOR TESTING
    model_config = {
        "json_schema_extra": {
            "examples": [
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
            ]
        }
    }

# 4. ENDPOINT PREDIKSI (SANG PELAYAN RESTORAN)
@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # A. Mengubah data JSON (Pydantic) menjadi format tabel (Pandas DataFrame)
        # Bawaan Pydantic v2 menyarankan model_dump(), tapi dict() masih aman di banyak versi
        df_input = pd.DataFrame([data.dict()])

        # B. PROSES TRANSLATE (Menggunakan Encoder dari Tahap 1)
        for col, le in encoders.items():
            if col in df_input.columns:
                # Keamanan ekstra: Jika user menginput teks yang belum pernah dipelajari AI (Misal typo)
                input_value = df_input[col][0]
                if input_value not in le.classes_:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Input tidak valid untuk kolom '{col}'. Pilihan yang benar: {list(le.classes_)}"
                    )
                # Terjemahkan teks menjadi angka
                df_input[col] = le.transform(df_input[col])

        # C. PASTIKAN URUTAN KOLOM SAMA PERSIS DENGAN SAAT TRAINING
        # Ini mencegah error karena urutan JSON dari user kadang acak
        df_input = df_input[features]

        # D. PREDIKSI MENGGUNAKAN XGBOOST
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)[0][1] # Ambil peluang untuk Churn (Label 1)

        # E. BUNGKUS JAWABAN (JSON RESPONSE)
        hasil_teks = "Churn (Akan Kabur)" if prediction[0] == 1 else "Retained (Setia)"
        
        return {
            "status": "success",
            "prediction_code": int(prediction[0]),
            "prediction_label": hasil_teks,
            "churn_probability": round(float(probability) * 100, 2), # Persentase
            "message": f"Sistem AI yakin {round(float(probability) * 100, 2)}% pelanggan ini akan kabur."
        }

    except Exception as e:
        # Menangkap error dari dalam sistem dan mengembalikannya secara elegan ke user
        raise HTTPException(status_code=500, detail=str(e))