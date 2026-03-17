import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

def train_model():
    print("=== TAHAP 1: ML PIPELINE (TRAINING & EVALUATION) ===")
    
    # 1. LOAD DATA
    path_data = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(path_data):
        print(f"❌ ERROR: File data tidak ditemukan di {path_data}")
        return
    df = pd.read_csv(path_data)

    # 2. DATA CLEANING
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop('customerID', axis=1)

    # 3. ENCODER
    print("⚙️ Menerjemahkan data teks menjadi angka...")
    encoders_dict = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders_dict[col] = le 

    # 4. SPLIT DATA
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    feature_columns = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. CLASS WEIGHTING & TRAINING XGBOOST
    rasio_bobot = sum(y_train == 0) / sum(y_train == 1)
    print("🧠 Melatih Model XGBoost dengan Penalti Ekstra (Class Weighting)...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=rasio_bobot)
    model.fit(X_train, y_train)

    # 6. EVALUASI DAN SIMPAN REPORT KE TXT
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    y_pred = model.predict(X_test)
    report_text = classification_report(y_test, y_pred)
    
    print("\n📊 Laporan Klasifikasi:")
    print(report_text)
    
    # Menyimpan teks ke dalam file
    with open('visualizations/8_classification_report.txt', 'w') as f:
        f.write("=== LAPORAN EVALUASI MODEL XGBOOST ===\n")
        f.write(f"Metode: Class Weighting (scale_pos_weight = {rasio_bobot:.2f})\n\n")
        f.write(report_text)
    print("✅ Laporan Klasifikasi berhasil diekspor ke 'visualizations/8_classification_report.txt'")

    # 7. VISUALISASI HASIL EVALUASI MODEL
    print("📈 Menggambar visualisasi hasil evaluasi AI...")

    # A. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Setia (0)', 'Kabur (1)'], yticklabels=['Setia (0)', 'Kabur (1)'])
    plt.ylabel('Fakta Sebenarnya (Actual)')
    plt.xlabel('Tebakan AI (Predicted)')
    plt.title('Confusion Matrix XGBoost')
    plt.savefig('visualizations/6_eval_confusion_matrix.png')
    plt.close()

    # B. Feature Importance
    importances = model.feature_importances_
    df_importances = pd.DataFrame({'Fitur': feature_columns, 'Pentingnya': importances})
    df_importances = df_importances.sort_values(by='Pentingnya', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_importances, x='Pentingnya', y='Fitur', palette='magma')
    plt.title('Top 10 Alasan AI Memprediksi Pelanggan Kabur')
    plt.savefig('visualizations/7_eval_feature_importance.png')
    plt.close()

    # 8. SERIALIZATION (SAVE MODEL)
    print("❄️ Membekukan Model dan Aset...")
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/churn_xgb_model.pkl')
    joblib.dump(encoders_dict, 'models/churn_encoders.pkl')
    joblib.dump(feature_columns, 'models/churn_features.pkl')
    print("✅ SUKSES! Model dan aset telah disimpan di folder 'models/'.")

if __name__ == "__main__":
    train_model()