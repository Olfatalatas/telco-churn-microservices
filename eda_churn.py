import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("=== TAHAP ANALISIS DATA: EXPLORATORY DATA ANALYSIS (EDA) ===")
    
    # 1. Siapkan Folder Output
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
        
    # 2. Load Data
    path_data = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(path_data):
        print(f"❌ ERROR: File data tidak ditemukan di {path_data}")
        return
        
    df = pd.read_csv(path_data)
    print(f"✅ Data berhasil diload. Total baris: {len(df)}")

    # 3. Data Cleaning (Hanya untuk keperluan visualisasi)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    print("📈 Sedang menggambar grafik analisis bisnis...")

    # Gambar 1: Distribusi Target
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Distribusi Pelanggan (Setia vs Kabur)')
    plt.savefig('visualizations/1_eda_churn_distribution.png')
    plt.close()

    # Gambar 2: Pengaruh Tipe Kontrak
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set1')
    plt.title('Tingkat Churn Berdasarkan Tipe Kontrak')
    plt.savefig('visualizations/2_eda_contract_vs_churn.png')
    plt.close()

    # Gambar 3: Pengaruh Lama Berlangganan (Tenure)
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple="stack", bins=36, palette='viridis')
    plt.title('Lama Berlangganan (Bulan) vs Tingkat Kabur')
    plt.xlabel('Tenure (Bulan)')
    plt.savefig('visualizations/3_eda_tenure_vs_churn.png')
    plt.close()

    # Gambar 4: Pengaruh Tagihan Bulanan (Monthly Charges)
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette='coolwarm')
    plt.title('Distribusi Tagihan Bulanan (Setia vs Kabur)')
    plt.xlabel('Tagihan Bulanan ($)')
    plt.savefig('visualizations/4_eda_monthly_charges.png')
    plt.close()

    # =========================================================
    # GAMBAR 5 BARU: PENGARUH GENDER TERHADAP CHURN
    # =========================================================
    plt.figure(figsize=(7, 5))
    # Kita buat bar plot untuk membandingkan jumlah Male dan Female yang Churn
    sns.countplot(data=df, x='gender', hue='Churn', palette='Pastel1')
    plt.title('Distribusi Kabur (Churn) Berdasarkan Gender')
    plt.ylabel('Jumlah Pelanggan')
    plt.xlabel('Gender')
    plt.savefig('visualizations/5_eda_gender_vs_churn.png')
    plt.close()

    print("✅ SUKSES! 5 Gambar EDA telah disimpan di folder 'visualizations/'.")

if __name__ == "__main__":
    run_eda()