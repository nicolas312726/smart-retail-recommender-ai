import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Smart Product Recommender", page_icon="🛍️", layout="centered")

st.title("🛍️ AI Product Recommendation System")
st.markdown("Aplikasi web ini merekomendasikan produk serupa berdasarkan kesamaan fitur deskripsi produk menggunakan **TF-IDF** dan **Cosine Similarity**.")
st.write("---")

# 2. Dataset Produk (Load Data)
@st.cache_data
def load_data():
    data_produk = {
        'id_produk': [1, 2, 3, 4, 5, 6],
        'nama_produk': [
            'Mouse Gaming Logitech G Pro Wireless',
            'Keyboard Mekanikal Razer BlackWidow',
            'Mouse Gaming Asus ROG Gladius',
            'Kursi Kerja Ergonomis Minimalis',
            'Keyboard Wireless Logitech MX Keys',
            'Meja Belajar Kayu Minimalis Lift-Up'
        ],
        'kategori': ['Gaming', 'Gaming', 'Gaming', 'Furniture', 'Office', 'Furniture'],
        'deskripsi': [
            'Mouse gaming tanpa kabel super ringan dengan sensor hero presisi tinggi untuk para atlet esport.',
            'Keyboard mekanikal RGB dengan switch responsif, cocok untuk setup gaming profesional.',
            'Mouse gaming kabel dengan desain ergonomis, sensor optik tajam, dan pencahayaan aura sync.',
            'Kursi kantor dengan sandaran punggung ergonomis untuk kenyamanan kerja berjam-jam.',
            'Keyboard wireless premium untuk produktivitas kerja mengetik cepat dan senyap multitarget.',
            'Meja kayu minimalis modern dengan fitur lift-up, sangat cocok untuk ruang kerja atau belajar.'
        ]
    }
    return pd.DataFrame(data_produk)

df = load_data()

# 3. Pemrosesan Kecerdasan Buatan (AI Engine)
# Membuat daftar kata hubung bawaan Indonesia secara manual
stop_words_indo = [
    'yang', 'di', 'dan', 'untuk', 'dengan', 'ini', 'itu', 'atau', 'ke', 'dari', 
    'adalah', 'bisa', 'akan', 'ada', 'perlu', 'kamu', 'anda', 'dalam', 'pada', 
    'juga', 'tetapi', 'jika', 'dia', 'secara', 'oleh', 'sangat', 'para'
]

# Menggunakan daftar kata hubung buatan kita
tfidf = TfidfVectorizer(stop_words=stop_words_indo)
tfidf_matrix = tfidf.fit_transform(df['deskripsi'])
skor_kemiripan = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Antarmuka Pengguna (User Interface)
st.subheader("🛒 Simulasi Pembelian Produk")
produk_dipilih = st.selectbox("Simulasikan pengguna sedang melihat atau membeli produk ini:", df['nama_produk'])

# Temukan indeks produk yang dipilih pengguna
idx_produk = df[df['nama_produk'] == produk_dipilih].index[0]

# Tampilkan detail produk utama yang sedang dilihat
st.info(f"**Deskripsi Produk yang Dilihat:**\n\n{df.loc[idx_produk, 'deskripsi']}")

st.write("---")

# 5. Mesin Rekomendasi (Recommendation Logic)
st.subheader("🔥 Rekomendasi Produk Terkait untuk Anda")

# Mengambil skor kemiripan produk tersebut dengan semua produk, lalu urutkan dari yang termirip
skor_produk = list(enumerate(skor_kemiripan[idx_produk]))
skor_produk = sorted(skor_produk, key=lambda x: x[1], reverse=True)

# Ambil 2 produk termirip (indeks ke-0 dibuang karena itu adalah dirinya sendiri)
rekomendasi_idx = [i[0] for i in skor_produk[1:3]] 

# Tampilkan hasil rekomendasi dalam bentuk kolom visual yang rapi
col1, col2 = st.columns(2)

with col1:
    st.image("https://placehold.co/300x200?text=Produk+Terkait", use_container_width=True)
    st.subheader(df.loc[rekomendasi_idx[0], 'nama_produk'])
    st.caption(f"Kategori: {df.loc[rekomendasi_idx[0], 'kategori']}")
    st.write(df.loc[rekomendasi_idx[0], 'deskripsi'])

with col2:
    st.image("https://placehold.co/300x200?text=Produk+Terkait", use_container_width=True)
    st.subheader(df.loc[rekomendasi_idx[1], 'nama_produk'])
    st.caption(f"Kategori: {df.loc[rekomendasi_idx[1], 'kategori']}")
    st.write(df.loc[rekomendasi_idx[1], 'deskripsi'])