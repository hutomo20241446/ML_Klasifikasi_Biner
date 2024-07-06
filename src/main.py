import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pickle

from ml_utility import (read_data, preprocess_data, train_model, evaluate_model)

# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)

st.set_page_config(page_title="Uas_Pak_Ayat", page_icon="🧠", layout="centered")

st.title("🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠")
st.title("Machine Learning Model Klasifikasi")
# Tambahkan dua tab
tab1, tab2 = st.tabs(["Pengantar", "Model Training & Prediksi"])

with tab1:
    st.write("""
        Kriteria Dataset untuk Pelatihan
        1. Label Biner: Dataset memiliki label yang bersifat biner (0 atau 1, iya atau tidak, dll).
        2. Fitur Numerik dan Kategorikal: Fitur kategorikal harus diubah menjadi numerik menggunakan teknik encoding. Fitur numerik sebaiknya diskalakan menggunakan scaler agar memiliki rentang nilai yang sebanding.
        3. Tidak Ada Missing Values: Data tidak boleh mengandung missing values. Jika ada, missing values harus ditangani sebelum melatih model. (misalnya, dengan imputasi).
        4. Distribusi Kelas: Distribusi kelas dalam label harus seimbang. Jika tidak, perlu dilakukan penanganan oversampling atau undersampling.
        5. Multikolinearitas: Fitur-fitur dalam dataset tidak memiliki multikolinearitas tinggi.
    """)
    
    st.markdown("""
        ### Petunjuk Penggunaan Aplikasi

        **1. Baca Kriteria Dataset:**
        - Sebelum memulai, pastikan dataset Anda memenuhi kriteria yang telah disebutkan di bagian pengantar, yaitu label biner, fitur numerik dan kategorikal, tidak ada missing values, distribusi kelas yang seimbang, dan tidak ada multikolinearitas tinggi.

        **2. Unggah Dataset:**
        - Klik pada tab "Model Training & Prediksi".
        - Unggah dataset Anda dengan cara klik tombol "Browse files" atau gunakan dataset sampel yang disediakan melalui link yang tersedia, Anda download terlebih dahulu kemudian unggah dataset yang didownload.
          Dataset harus dalam format CSV, XLS, atau XLSX.

        **3. Pilih Target dan Konfigurasi Pelatihan:**
        - Setelah dataset diunggah, dataset akan ditampilkan dalam bentuk tabel.
        - Pilih kolom target (label) dari dropdown "Pilih Target".
        - Pilih jenis scaler dari dropdown "Pilih scaler" (standard atau minmax).
        - Pilih model machine learning dari dropdown "Pilih Model" (Logistic Regression, SVM, Random Forest, XGBoost, Gaussian Naive Bayes, Decision Tree).
        - Masukkan nama model yang akan dilatih di kolom "Nama Model".

        **4. Latih Model:**
        - Klik tombol "Latih Model" untuk memulai proses pelatihan.
        - Aplikasi akan memproses data, melatih model, dan menampilkan akurasi model setelah pelatihan selesai.
        - Model yang dilatih akan tersimpan selama sesi aktif di browser Anda.

        **5. Lakukan Prediksi:**
        - Setelah model dilatih, Anda dapat menggunakan model untuk melakukan prediksi.
        - Pilih model yang sudah dilatih dari dropdown "Pilih Model untuk Memprediksi".
        - Masukkan data baru melalui formulir input yang disediakan untuk setiap fitur.
        - tekan "Enter" setiap kali menginput data baru ke formulir.
        - Klik tombol "Prediksi" untuk mendapatkan hasil prediksi.

        **6. Lihat Daftar Model yang Tersimpan:**
        - Daftar model yang sudah dilatih dan informasi akurasinya dapat dilihat di tabel "Daftar Model yang Tersimpan".
        - Informasi ini membantu Anda membandingkan performa model yang berbeda dan memilih model terbaik untuk digunakan.

        Silahkan email ke [mfhutomo@gmail.com](mailto:mfhutomo@gmail.com) untuk bantuan lebih lanjut. 
    """)
    st.write('Source code dapat diakses [di sini](https://github.com/hutomo20241446/ML_Klasifikasi_Biner/blob/main/src)')

# Tab 2
with tab2:
    # Delete trained models on refresh
    trained_model_dir = f"{parent_dir}/trainedmodel"
    if 'init' not in st.session_state:
        st.session_state['init'] = True
        if os.path.exists(trained_model_dir):
            for file in os.listdir(trained_model_dir):
                file_path = os.path.join(trained_model_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    # Menampilkan widget untuk upload file
    uploaded_file = st.file_uploader("Silahkan upload dataset atau gunakan dataset sampel [di sini](https://drive.google.com/drive/folders/1uxk1bVQIV1p2Jfs9AU_Pk4o2wNUNRlaW?usp=sharing)", type=["csv", "xls", "xlsx"])

    # Fungsi untuk membaca data
    def read_data(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return df
        elif file.name.endswith('.xls') or file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
            return df
        else:
            st.write("Format file tidak didukung")

    # Memproses file yang diunggah
    df = None
    if uploaded_file is not None:
        df = read_data(uploaded_file)
        if df is not None:
            st.write(df)
        else:
            st.write("Format file tidak didukung")

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)

        scaler_type_list = ["standard", "minmax"]

        model_dictionary = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": XGBClassifier(),
            "Gaussian Naive Bayes": GaussianNB(),
            "Decision Tree Classifier": DecisionTreeClassifier()
        }

        with col1:
            target_column = st.selectbox("Pilih Target", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Pilih scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Pilih Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Nama Model")

        if st.button("Latih Model"):
            X_train, X_test, y_train, y_test, preprocessor, target_mapping = preprocess_data(df, target_column, scaler_type)
            model_to_be_trained = model_dictionary[selected_model]
            model = train_model(X_train, y_train, preprocessor, model_to_be_trained, model_name)
            accuracy = evaluate_model(model, X_test, y_test)

            # Simpan informasi model
            model_info = {"Model Name": model_name, "Target": target_column, "Scaler": scaler_type, "Selected Model": selected_model, "Accuracy": accuracy, "Target Mapping": target_mapping}
            model_info_path = f"{trained_model_dir}/model_info.csv"
            if os.path.exists(model_info_path):
                model_info_df = pd.read_csv(model_info_path)
            else:
                model_info_df = pd.DataFrame(columns=["Model Name", "Target", "Scaler", "Selected Model", "Accuracy", "Target Mapping"])

            # Hapus kolom atau baris yang semuanya ber-NA
            model_info_df.dropna(how='all', axis=1, inplace=True)

            model_info_df = pd.concat([model_info_df, pd.DataFrame([model_info])], ignore_index=True)
            model_info_df.to_csv(model_info_path, index=False)

            st.success(f"Tes Akurasi: {accuracy}")

    # Menampilkan tabel informasi model
    model_info_path = f"{trained_model_dir}/model_info.csv"
    if os.path.exists(model_info_path):
        model_info_df = pd.read_csv(model_info_path)
        st.write("### Daftar Model yang Tersimpan")
        st.dataframe(model_info_df.drop(columns=["Target Mapping"]))

        # Pilih model yang disimpan untuk prediksi
        selected_saved_model = st.selectbox("Pilih Model untuk Memprediksi", model_info_df["Model Name"].unique())
        if selected_saved_model:
            model_file_path = f"{trained_model_dir}/{selected_saved_model}.pkl"
            if os.path.exists(model_file_path):
                with open(model_file_path, 'rb') as file:
                    model = pickle.load(file)

                # Mendapatkan target mapping dari informasi model yang disimpan
                target_mapping_str = model_info_df.loc[model_info_df["Model Name"] == selected_saved_model, "Target Mapping"].values[0]
                target_mapping = eval(target_mapping_str)  # Mengubah string kembali ke dictionary

                st.write(f"### Membuat prediksi menggunakan {selected_saved_model}")

                # Ambil kolom fitur dari dataframe yang diunggah
                feature_columns = df.drop(columns=[target_column]).columns.tolist()

                # Inisialisasi state input
                if 'input_data' not in st.session_state:
                    st.session_state.input_data = {col: '' for col in feature_columns}

                # Buat form untuk input data baru
                for col in feature_columns:
                    st.session_state.input_data[col] = st.text_input(f"Input {col}", st.session_state.input_data[col])

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Prediksi"):
                        new_data_df = pd.DataFrame([st.session_state.input_data])
                        new_data_df = new_data_df.apply(pd.to_numeric, errors='ignore')  # Convert to numeric where possible
                        prediction = model.predict(new_data_df)
                        st.write(f"Hasil Prediksi: {list(target_mapping.keys())[list(target_mapping.values()).index(prediction[0])]}")
                
                with col2:
                    if st.button("Hapus"):
                        for col in feature_columns:
                            st.session_state.input_data[col] = ''
                        st.experimental_rerun()
    else:
        st.write("Belum ada model yang tersimpan")


# Membuat footer
footer = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
    }
</style>
<div class="footer">
    <p>&copy;Muhammad Fahmi Hutomo</p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
