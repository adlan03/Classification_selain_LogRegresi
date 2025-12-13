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

def prepare_features(df: pd.DataFrame, drop_cols, target_col):
    df = df.copy()

    needed_cols = {"Order Date", "Ship Date"}
    if not needed_cols.issubset(df.columns):
        missing = needed_cols - set(df.columns)
        raise ValueError(f"Kolom wajib tidak ada: {', '.join(missing)}")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"], errors="coerce")

    df["OrderYear"]  = df["Order Date"].dt.year
    df["OrderMonth"] = df["Order Date"].dt.month
    df["ShipDays"]   = (df["Ship Date"] - df["Order Date"]).dt.days

    if df["ShipDays"].isna().any():
        df["ShipDays"] = df["ShipDays"].fillna(df["ShipDays"].median())

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df

# =========================
# UI: HEADER
# =========================
st.title("📌 Aplikasi Prediksi Segment Pelanggan — Superstore")
st.markdown(
    """
Aplikasi ini dibuat untuk **memprediksi Segment pelanggan** pada data Superstore (**Consumer / Corporate / Home Office**).
Tujuannya: membantu memahami **tipe pelanggan** dari transaksi, sehingga analisis dan strategi pemasaran lebih tepat.
"""
)

# =========================
# SIDEBAR: PATH FILE (tanpa upload)
# =========================
st.sidebar.header("⚙️ Sumber File (tanpa upload)")
pkl_path = st.sidebar.text_input("File model (.pkl)", "superstore_segment_voting.pkl")
csv_path = st.sidebar.text_input("File dataset (.csv)", "Sample - Superstore.csv")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model")
st.sidebar.write(
    """
✅ Ensemble: **Voting Classifier**  
✅ Tanpa **DT**, **KNN**, **Logistic Regression**  
"""
)

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
# TAMPILKAN DATASET
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Baris", f"{df_raw.shape[0]:,}")
c2.metric("Jumlah Kolom", f"{df_raw.shape[1]}")
c3.metric("Target", "Segment")

st.subheader("🔎 Preview Dataset (20 baris pertama)")
st.dataframe(df_raw.head(20), use_container_width=True)

# =========================
# PREDIKSI
# =========================
st.markdown("---")
st.subheader("🚀 Jalankan Prediksi")

if st.button("Prediksi Segment", type="primary"):
    try:
        X_pred = prepare_features(df_raw, drop_cols, target_col)
        preds = model.predict(X_pred)

        out = df_raw.copy()
        out["Predicted Segment"] = preds

        tab1, tab2, tab3 = st.tabs(["📌 Ringkasan", "📄 Tabel Hasil", "⬇️ Download"])

        with tab1:
            st.subheader("Distribusi Hasil Prediksi")
            st.write(out["Predicted Segment"].value_counts())

            st.markdown(
                """
**Cara baca ringkasan:**
- Angka menunjukkan **jumlah transaksi** yang diprediksi masuk ke tiap segment.
- Ini membantu melihat **segment dominan** pada data Superstore.
"""
            )

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
