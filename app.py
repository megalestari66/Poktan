import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Fungsi untuk mengisi nilai yang hilang
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == object:
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    return df

# Fungsi untuk membuat plot hasil prediksi vs nilai aktual
def plot_prediction_vs_actual(y_true, y_pred, score):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue', label='Prediksi')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Garis Ideal')
    plt.xlabel("Nilai Aktual")
    plt.ylabel("Nilai Prediksi")
    plt.title(f"Prediksi vs Aktual (R² Score: {score:.2f})")
    plt.legend()
    st.pyplot(plt)

# Fungsi untuk membuat plot distribusi error (residuals)
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Nilai Prediksi")
    plt.ylabel("Residuals")
    plt.title("Plot Residuals")
    st.pyplot(plt)

# Fungsi untuk mengembalikan label berdasarkan prediksi
def get_category(pred):
    if pred <= 245:
        return 'Pemula'
    elif pred <= 455:
        return 'Lanjut'
    elif pred <= 700:
        return 'Madya'
    else:
        return 'Utama'

def get_value(label):
    if label == "Pemula":
        return 0
    elif label == "Lanjut":
        return 1
    elif label == "Madya":
        return 2
    else: 
        return 3

# Fungsi untuk menyoroti kolom atribut dan label di menu Atribut Label
def highlight_columns(x):
    attr_style = 'background-color: #FFFACD;'  # Mengganti dengan warna #E9967A
    label_style = 'background-color: grey; color: black'
    
    df = pd.DataFrame('', index=x.index, columns=x.columns)
    df.iloc[:, :-1] = attr_style
    df.iloc[:, -1] = label_style
    return df

# Fungsi untuk menyoroti kolom di menu Dataset
def highlight_columns_dataset(x):
    attr_style = 'background-color: #FFFACD;'  # Mengganti dengan warna #E9967A
    
    df = pd.DataFrame('', index=x.index, columns=x.columns)
    df.iloc[:, :] = attr_style
    return df

# Fungsi untuk melatih model dan mengembalikan model serta skor
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    return model, score

# Fungsi untuk melakukan prediksi
def predict(model, label_encoders, input_df):
    for column in input_df.columns:
        if column in label_encoders:
            input_df[column] = label_encoders[column].transform(input_df[column])
    pred = model.predict(input_df)
    return pred

#Inisiasi layout dari streamlite
# Sidebar navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Menu", ["Beranda", "Regresi Linear"])

if menu == "Beranda":
    st.markdown("<h1 style='text-align: center;'>Prediksi Penilaian Kemampuan Kelompok Tani</h1>", unsafe_allow_html=True)
    st.image("dinas.png", caption="Data Analysis", use_column_width=True)
    st.markdown("""
    ## Deskripsi Perusahaan
    Kelompok tani merupakan organisasi yang bekerja sebagai petani dan berkumpul dalam suatu kelompok dengan tujuan dan minat yang sama pemerintah, lembaga pertanian, dan kelompok tani sendiri dapat menggunakan penilaian kelompok tani untuk membuat keputusan tentang alokasi sumber daya, pelatihan, dan strategi pengembangan pertanian yang lebih sesuai dengan pembagian kelas. Dinas Tanaman Pangan Hortikultura dan Tanaman Perkebunan Balai Penyuluh Pertanian (BPP) Ketapang sebagai pemerintahan yang berfokus pada pertanian harus mampu meningkatkan setiap bidang yang ada agar lebih baik lagi, khususnya pada bagian penilaian kemampuan kelompok tani. Untuk mengatasi masalah ini, penelitian ini mengusulkan untuk menerapkan metode Regresi Linear upaya memprediksi penilaian kemampuan kelompok tani.
    
    ## Tentang Kami
    Lokasi: Balai Penyuluhan Pertanian (BPP) Ketapang, Jln. Ikan Tongkol No. 16 Desa Ketapang, Lampung Selatan.
    
    Untuk pertanyaan atau dukungan apa pun, silakan hubungi [email: bp3k.ketapang@gmail.com]
    """)

elif menu == "Regresi Linear":
    submenu = st.sidebar.radio("Submenu Regresi Linear", ["Dataset", "Atribut Label", "Prediksi"])

    if submenu == "Dataset":
        st.title("Dataset")
        st.write("Unggah dan kelola dataset Anda di sini.")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Menangani kesalahan saat membaca CSV
                df = pd.read_csv(uploaded_file, on_bad_lines='skip', sep=',')
                st.session_state.df = df

                if 'Total' in df.columns:
                    st.session_state.df_display = df.drop(columns=['Total'])
                else:
                    st.session_state.df_display = df
                
                st.write("Dataset")
                st.dataframe(st.session_state.df_display.style.apply(highlight_columns_dataset, axis=None))

                # Tambahkan label berdasarkan 'Total' jika belum ada
                if 'Label' not in df.columns:
                    df['Label'] = df['Total'].apply(lambda x: 'Pemula' if x <= 245 else 'Lanjut' if x <= 455 else 'Madya' if x <= 700 else 'Utama')
            except pd.errors.ParserError as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")

    elif submenu == "Atribut Label":
        st.title("Atribut Label")
        st.write("Tentukan atribut dan label untuk dataset Anda.")
        
        if 'df' in st.session_state:
            df = st.session_state.df.drop(columns=['alamat','nik_ketua','ketua_kelompok','no_telp','tahun_pembentukan','poktan_l','poktan_p','nama_penyuluh'], errors='ignore')
            st.session_state.df = fill_missing_values(df)

            df_with_total_and_label = st.session_state.df.copy()
            if 'Label' not in df_with_total_and_label.columns:
                df_with_total_and_label['Label'] = df_with_total_and_label['Total'].apply(lambda x: 'Pemula' if x <= 245 else 'Lanjut' if x <= 455 else 'Madya' if x <= 700 else 'Utama')
            
            st.write("Dataset dengan atribut dan label yang disorot")
            st.dataframe(df_with_total_and_label.style.apply(highlight_columns, axis=None))
            
            st.write("Filter berdasarkan kelas:")
            kelas = st.selectbox("Pilih kelas", ["Semua", "Pemula", "Lanjut", "Madya", "Utama"])
            
            if kelas != "Semua":
                kelas_dict = {"Pemula": "Pemula", "Lanjut": 'Lanjut', "Madya": 'Madya', "Utama": 'Utama'}
                filtered_df = df_with_total_and_label[df_with_total_and_label['Label'] == kelas_dict[kelas]]
                st.write(f"Dataset - Kelas: {kelas}")
                st.dataframe(filtered_df.style.apply(highlight_columns, axis=None))
            else:
                st.write("Dataset - Semua Kelas")
                st.dataframe(df_with_total_and_label.style.apply(highlight_columns, axis=None))

            st.write("Visualisasi Distribusi Kelas")
            label_counts = df_with_total_and_label['Label'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis', ax=ax)
            ax.set_xlabel('Kelas')
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Kelas (Label)')
            st.pyplot(fig)

            st.write("Korelasi Antar Fitur")
            corr = df_with_total_and_label.drop(columns=['Label', 'nama_desa', 'nama_kelompok']).corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
            plt.title("Matriks Korelasi")
            st.pyplot(plt)

        else:
            st.warning("Harap unggah dataset terlebih dahulu di menu Dataset.")
    elif submenu == "Prediksi":
        st.title("Prediksi")
        st.write("Buat prediksi berdasarkan model yang telah dilatih di sini.")
        
        if 'df' in st.session_state:
            df = st.session_state.df.copy()
            if 'Label' not in df.columns:
                df['Label'] = df['Total'].apply(lambda x: 0 if x <= 245 else 1 if x <= 455 else 2 if x <= 700 else 3)
            X = df.drop(columns=['Label'])  # Semua kolom kecuali 'Label'
            Z = df.drop(columns=['Label','nama_desa', 'nama_kelompok'])  # Semua kolom kecuali 'Label'
            y = df['Label'].apply(get_value)   # Kolom 'Label'

            # Inisialisasi dan simpan label encoder
            label_encoders = {}
            for column in Z.columns:
                if X[column].dtype == object:
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column])
                    label_encoders[column] = le

            st.session_state.label_encoders = label_encoders

            # Latih model dan simpan di session state
            model, score = train_model(Z, y)
            st.session_state.clf = model
            st.session_state.score = score

            # Input manual untuk prediksi
            st.write("Masukkan nilai untuk prediksi:")
            col1, col2, col3 = st.columns(3)
            user_input = {}
            #Iterasi kolom dari fitur untuk dijadikan nama fitur pada kolom input
            for column in X.columns:
                if column == 'nama_desa' or column == 'nama_kelompok':
                    if len(user_input) % 3 == 0:
                        user_input[column] = col1.text_input(column)
                    elif len(user_input) % 3 == 1:
                        user_input[column] = col2.text_input(column)
                    else:
                        user_input[column] = col3.text_input(column)
                else:
                    if len(user_input) % 3 == 0:
                        user_input[column] = col1.number_input(column, value=0)
                    elif len(user_input) % 3 == 1:
                        user_input[column] = col2.number_input(column, value=0)
                    else:
                        user_input[column] = col3.number_input(column, value=0)

            if st.button("Prediksi"):

                process_data = user_input
                process_encoders = label_encoders
                desa = process_data['nama_desa']
                kelompok = process_data['nama_kelompok']
                del process_data['nama_desa']
                del process_data['nama_kelompok']
                input_df = pd.DataFrame([process_data])              
                # Lakukan prediksi
                pred = predict(model, process_encoders, input_df)
                # Konversi prediksi menjadi label kategori
                target = get_category(process_data['Total'])  # Ambil nilai prediksi pertama
                
                # Tampilkan hasil prediksi dalam kolom
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)
              
                col1.write("Hasil")
                col2.write(f"Nama Desa: {desa}")
                col3.write(f"Nama Kelompok: {kelompok}")
                col4.write(f"Total: {process_data['Total']}")
                col5.write(f"Prediksi: {target}")
                col6.write(f"R² Score: {score:.2f}")
                
                # Visualisasi hasil prediksi
                st.write("Visualisasi Prediksi vs Nilai Aktual")
                plot_prediction_vs_actual(y, model.predict(Z), score)

                # Visualisasi tambahan
                st.write("Distribusi Error (Residuals)")
                plot_residuals(y, model.predict(Z))


# Tampilkan konten utama untuk setiap submenu
if menu != "Beranda":
    st.markdown(
        """
        <style>
        .content {
            margin-top: 20px;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
