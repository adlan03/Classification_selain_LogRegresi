import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Prediksi Segment Superstore", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)

# =========================
# LOAD DATASET (langsung dari file)
# =========================
@st.cache_data
def load_dataset(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {csv_path}")
    try:
        return pd.read_csv(p, encoding="latin1")
    except Exception:
        return pd.read_csv(p)

# =========================
# FEATURE ENGINEERING (tampilkan di Streamlit)
# =========================
def add_feature_engineering(df: pd.DataFrame):
    df = df.copy()

    needed_cols = {"Order Date", "Ship Date"}
    if not needed_cols.issubset(df.columns):
        missing = needed_cols - set(df.columns)
        raise ValueError(f"Kolom wajib tidak ada: {', '.join(missing)}")

    # convert date
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"], errors="coerce")

    # engineered features
    df["OrderYear"]  = df["Order Date"].dt.year
    df["OrderMonth"] = df["Order Date"].dt.month
    df["ShipDays"]   = (df["Ship Date"] - df["Order Date"]).dt.days

    # handle NaN shipdays
    if df["ShipDays"].isna().any():
        df["ShipDays"] = df["ShipDays"].fillna(df["ShipDays"].median())

    return df

def prepare_features(df_enriched: pd.DataFrame, drop_cols, target_col):
    df = df_enriched.copy()

    # drop kolom sesuai training
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # jika ada label asli, buang (mode prediksi)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df

# =========================
# UI HEADER
# =========================
st.title("📌 Aplikasi Prediksi Segment Pelanggan — Superstore")
st.markdown(
    """
Aplikasi ini memprediksi **Segment pelanggan** pada Superstore (**Consumer / Corporate / Home Office**).
Model sudah dilatih sebelumnya dan disimpan sebagai **PKL**, jadi di aplikasi ini **tidak training lagi** — hanya **load + predict**.
"""
)

# =========================
# SIDEBAR PATH
# =========================
st.sidebar.header("⚙️ Sumber File (tanpa upload)")
pkl_path = st.sidebar.text_input("File model (.pkl)", "superstore_segment_voting.pkl")
csv_path = st.sidebar.text_input("File dataset (.csv)", "Sample - Superstore.csv")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model")
st.sidebar.write("✅ Ensemble: Voting Classifier")
st.sidebar.write("✅ Tanpa DT, tanpa KNN, tanpa Logistic Regression")

# Load model
try:
    artifact = load_artifact(pkl_path)
    model = artifact["model"]
    drop_cols = artifact["drop_cols"]
    target_col = artifact["target_col"]
    st.sidebar.success("Model PKL berhasil diload ✅")
except Exception as e:
    st.sidebar.error(f"Gagal load PKL: {e}")
    st.stop()

# Load dataset
try:
    df_raw = load_dataset(csv_path)
except Exception as e:
    st.error(f"❌ Gagal load dataset: {e}")
    st.info("Pastikan file CSV ada di folder yang sama dengan main.py, atau ubah path di sidebar.")
    st.stop()

# =========================
# DATASET OVERVIEW
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Baris", f"{df_raw.shape[0]:,}")
c2.metric("Jumlah Kolom", f"{df_raw.shape[1]}")
c3.metric("Target", "Segment")

st.subheader("🔎 Preview Dataset Asli (sebelum Feature Engineering)")
st.dataframe(df_raw.head(15), use_container_width=True)

# =========================
# FEATURE ENGINEERING DISPLAY
# =========================
st.markdown("---")
st.header("🛠️ Feature Engineering yang Dipakai")

st.markdown(
    """
Di tahap ini, kolom tanggal **Order Date** dan **Ship Date** diubah menjadi fitur yang lebih “siap” dipakai model:
- **OrderYear**: tahun pemesanan  
- **OrderMonth**: bulan pemesanan  
- **ShipDays**: selisih hari antara pengiriman dan pemesanan  
"""
)

try:
    df_enriched = add_feature_engineering(df_raw)

    # tampilkan hasil feature engineering (kolom penting)
    st.subheader("Hasil Feature Engineering (contoh 20 baris)")
    show_cols = [c for c in ["Order Date", "Ship Date", "OrderYear", "OrderMonth", "ShipDays"] if c in df_enriched.columns]
    st.dataframe(df_enriched[show_cols].head(20), use_container_width=True)

    # ringkasan statistik ShipDays (biar mudah dipresentasi)
    st.subheader("Ringkasan ShipDays")
    st.write(df_enriched["ShipDays"].describe())

    # optional: tampilkan list fitur yang dipakai model setelah drop
    with st.expander("Lihat kolom input yang dipakai model (setelah drop kolom ID)"):
        X_for_model = prepare_features(df_enriched, drop_cols, target_col)
        st.write("Jumlah kolom input ke model:", X_for_model.shape[1])
        st.write(list(X_for_model.columns))

except Exception as e:
    st.error(f"❌ Feature engineering gagal: {e}")
    st.stop()

# =========================
# PREDIKSI
# =========================
st.markdown("---")
st.header("🚀 Prediksi Segment")

if st.button("Prediksi Segment", type="primary"):
    try:
        X_pred = prepare_features(df_enriched, drop_cols, target_col)
        preds = model.predict(X_pred)

        out = df_raw.copy()
        out["Predicted Segment"] = preds

        tab1, tab2, tab3 = st.tabs(["📌 Ringkasan", "📄 Tabel Hasil", "⬇️ Download"])

        with tab1:
            st.subheader("Distribusi Hasil Prediksi")
            st.write(out["Predicted Segment"].value_counts())

        with tab2:
            st.subheader("Preview Hasil Prediksi (50 baris)")
            st.dataframe(out.head(50), use_container_width=True)

        with tab3:
            st.subheader("Download Hasil Prediksi")
            st.download_button(
                "Download hasil prediksi (.csv)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="superstore_predicted_segment.csv",
                mime="text/csv"
            )

        st.success("✅ Prediksi selesai!")

    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")
