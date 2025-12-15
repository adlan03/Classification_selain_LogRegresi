import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.set_page_config(page_title="Superstore - Klasifikasi & Regresi", layout="wide")

# =========================
# LOAD MODEL & DATASET
# =========================
@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)

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
# FEATURE ENGINEERING
# =========================
def add_feature_engineering(df: pd.DataFrame):
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

    return df

def prepare_features(df_enriched: pd.DataFrame, drop_cols, target_col):
    df = df_enriched.copy()

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    return df

# =========================
# INPUT BUILDER (MANUAL 1 ROW)
# =========================
def build_manual_row_inputs(feature_cols, df_ref, order_date, ship_date):
    row = {}

    order_year = pd.Timestamp(order_date).year
    order_month = pd.Timestamp(order_date).month
    ship_days = (pd.Timestamp(ship_date) - pd.Timestamp(order_date)).days

    c1, c2, c3 = st.columns(3)
    c1.metric("OrderYear", order_year)
    c2.metric("OrderMonth", order_month)
    c3.metric("ShipDays", ship_days)

    st.markdown("### Isi atribut transaksi")

    for col in feature_cols:
        if col == "OrderYear":
            row[col] = int(order_year)
            continue
        if col == "OrderMonth":
            row[col] = int(order_month)
            continue
        if col == "ShipDays":
            row[col] = int(ship_days)
            continue

        if col in df_ref.columns:
            # numerik
            if pd.api.types.is_numeric_dtype(df_ref[col]):
                s = pd.to_numeric(df_ref[col], errors="coerce")
                mn = float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else 0.0
                mx = float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else 100.0
                med = float(np.nanmedian(s)) if np.isfinite(np.nanmedian(s)) else 0.0
                if mn == mx:
                    mn, mx = mn - 1, mx + 1
                row[col] = st.number_input(col, value=med, min_value=mn, max_value=mx)
            else:
                uniq = df_ref[col].dropna().astype(str).unique()
                nuniq = len(uniq)
                if nuniq <= 80:
                    row[col] = st.selectbox(col, sorted(list(uniq)))
                else:
                    default_val = str(df_ref[col].dropna().astype(str).iloc[0]) if df_ref[col].notna().any() else ""
                    row[col] = st.text_input(col, value=default_val)
        else:
            row[col] = st.text_input(col, value="")

    return row

# =========================
# HEADER
# =========================
st.title("📌 Aplikasi Superstore — Klasifikasi Segment & Regresi Sales")
st.markdown(
    """
Aplikasi ini untuk demo presentasi:
- **Klasifikasi Segment**: Consumer / Corporate / Home Office (PKL VotingClassifier)
- **Regresi Sales**: prediksi nilai Sales (PKL RandomForestRegressor)

Dataset dibaca langsung dari file CSV (tanpa upload) dan model dibaca dari file PKL.
"""
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Sumber File (tanpa upload)")
csv_path = st.sidebar.text_input("File dataset (.csv)", "Sample - Superstore.csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Model PKL")
pkl_cls_path = st.sidebar.text_input("PKL Klasifikasi Segment", "superstore_segment_voting.pkl")
pkl_reg_path = st.sidebar.text_input("PKL Regresi Sales (RF)", "superstore_reg_sales_rf.pkl")

# =========================
# LOAD DATASET
# =========================
try:
    df_raw = load_dataset(csv_path)
except Exception as e:
    st.error(f"❌ Gagal load dataset: {e}")
    st.info("Pastikan CSV ada di folder yang sama dengan main.py, atau ubah path di sidebar.")
    st.stop()

# overview dataset
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Baris", f"{df_raw.shape[0]:,}")
c2.metric("Jumlah Kolom", f"{df_raw.shape[1]}")
c3.metric("CSV", Path(csv_path).name)

st.subheader("🔎 Preview Dataset Asli (sebelum Feature Engineering)")
st.dataframe(df_raw.head(15), use_container_width=True)

# =========================
# FEATURE ENGINEERING
# =========================
st.markdown("---")
st.header("🛠️ Feature Engineering")
st.markdown(
    """
Kolom tanggal diubah jadi fitur:
- **OrderYear**
- **OrderMonth**
- **ShipDays**
"""
)

try:
    df_enriched = add_feature_engineering(df_raw)
    show_cols = [c for c in ["Order Date", "Ship Date", "OrderYear", "OrderMonth", "ShipDays"] if c in df_enriched.columns]
    st.subheader("Contoh hasil Feature Engineering (20 baris)")
    st.dataframe(df_enriched[show_cols].head(20), use_container_width=True)
    st.subheader("Ringkasan ShipDays")
    st.write(df_enriched["ShipDays"].describe())
except Exception as e:
    st.error(f"❌ Feature engineering gagal: {e}")
    st.stop()

# =========================
# TABS
# =========================
tab_cls, tab_reg = st.tabs(["🧠 Klasifikasi Segment", "📈 Regresi Sales"])

# ==========================================
# TAB KLASIFIKASI
# ==========================================
with tab_cls:
    st.subheader("Klasifikasi: Prediksi Segment Pelanggan")

    try:
        artifact_cls = load_artifact(pkl_cls_path)
        model_cls = artifact_cls["model"]
        drop_cols_cls = artifact_cls["drop_cols"]
        target_cls = artifact_cls["target_col"]  # "Segment"
        st.success("✅ PKL Klasifikasi berhasil diload")
    except Exception as e:
        st.error(f"❌ Gagal load PKL klasifikasi: {e}")
        st.stop()

    X_cols_cls = prepare_features(df_enriched, drop_cols_cls, target_cls).columns.tolist()

    with st.expander("Lihat kolom input yang dipakai model klasifikasi"):
        st.write("Jumlah kolom input:", len(X_cols_cls))
        st.write(X_cols_cls)

    sub_manual, sub_batch = st.tabs(["🧾 Input Manual (1 transaksi)", "📦 Prediksi Massal (CSV)"])

    # ---- Manual
    with sub_manual:
        st.markdown("Isi form ini untuk demo prediksi **1 transaksi**.")

        default_od = pd.to_datetime(df_raw["Order Date"], errors="coerce").dropna()
        default_sd = pd.to_datetime(df_raw["Ship Date"], errors="coerce").dropna()

        with st.form("form_cls_manual"):
            colA, colB = st.columns(2)
            od = colA.date_input("Order Date", value=default_od.iloc[0].date() if len(default_od) else pd.Timestamp.today().date())
            sd = colB.date_input("Ship Date", value=default_sd.iloc[0].date() if len(default_sd) else pd.Timestamp.today().date())

            row_dict = build_manual_row_inputs(X_cols_cls, df_enriched, od, sd)
            submitted = st.form_submit_button("Prediksi Segment", type="primary")

        if submitted:
            try:
                X_one = pd.DataFrame([row_dict])
                pred = model_cls.predict(X_one)[0]
                st.success(f"✅ Prediksi Segment: **{pred}**")
                st.dataframe(X_one, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Gagal prediksi manual: {e}")

    # ---- Batch + Visualisasi
    with sub_batch:
        st.markdown("Prediksi **seluruh baris** pada CSV yang sedang dibaca.")
        if st.button("Prediksi Segment (Massal)", type="primary"):
            try:
                X_pred = prepare_features(df_enriched, drop_cols_cls, target_cls)
                preds = model_cls.predict(X_pred)

                out = df_raw.copy()
                out["Predicted Segment"] = preds

                tab1, tab2, tab3, tab4 = st.tabs(["📌 Ringkasan", "📊 Visualisasi", "📄 Tabel", "⬇️ Download"])

                with tab1:
                    st.subheader("Distribusi Prediksi Segment")
                    st.write(out["Predicted Segment"].value_counts())

                with tab2:
                    st.subheader("📊 Visualisasi Klasifikasi")

                    # Bar chart distribusi prediksi
                    fig_bar, ax_bar = plt.subplots()
                    out["Predicted Segment"].value_counts().plot(kind="bar", ax=ax_bar)
                    ax_bar.set_title("Distribusi Predicted Segment")
                    ax_bar.set_xlabel("Segment")
                    ax_bar.set_ylabel("Jumlah")
                    st.pyplot(fig_bar)

                    # Confusion matrix & report kalau ada label asli
                    if target_cls in df_raw.columns:
                        y_true = df_raw[target_cls].astype(str)
                        y_pred = out["Predicted Segment"].astype(str)

                        acc = accuracy_score(y_true, y_pred)
                        st.write(f"**Accuracy:** {acc:.4f}")

                        labels = sorted(y_true.unique().tolist())
                        cm = confusion_matrix(y_true, y_pred, labels=labels)

                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt="d",
                                    xticklabels=labels, yticklabels=labels, ax=ax_cm)
                        ax_cm.set_title("Confusion Matrix (Count)")
                        ax_cm.set_xlabel("Predicted")
                        ax_cm.set_ylabel("Actual")
                        st.pyplot(fig_cm)

                        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                        fig_cmn, ax_cmn = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm_norm, annot=True, fmt=".2f",
                                    xticklabels=labels, yticklabels=labels, ax=ax_cmn)
                        ax_cmn.set_title("Confusion Matrix (Normalized)")
                        ax_cmn.set_xlabel("Predicted")
                        ax_cmn.set_ylabel("Actual")
                        st.pyplot(fig_cmn)

                        st.subheader("Classification Report")
                        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
                    else:
                        st.info("Kolom 'Segment' tidak ada di CSV, jadi confusion matrix & report tidak bisa dibuat.")

                with tab3:
                    st.subheader("Preview hasil (50 baris)")
                    st.dataframe(out.head(50), use_container_width=True)

                with tab4:
                    st.download_button(
                        "Download hasil segment (.csv)",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="superstore_predicted_segment.csv",
                        mime="text/csv"
                    )

                st.success("✅ Prediksi massal selesai!")
            except Exception as e:
                st.error(f"❌ Error prediksi massal: {e}")

# ==========================================
# TAB REGRESI
# ==========================================
with tab_reg:
    st.subheader("Regresi: Prediksi Sales (angka)")

    try:
        artifact_reg = load_artifact(pkl_reg_path)
        model_reg = artifact_reg["model"]
        drop_cols_reg = artifact_reg["drop_cols"]
        target_reg = artifact_reg["target_col"]  # "Sales"
        st.success("✅ PKL Regresi berhasil diload")
    except Exception as e:
        st.error(f"❌ Gagal load PKL regresi: {e}")
        st.stop()

    st.info(f"Target regresi dari PKL: **{target_reg}**")
    X_cols_reg = prepare_features(df_enriched, drop_cols_reg, target_reg).columns.tolist()

    with st.expander("Lihat kolom input yang dipakai model regresi"):
        st.write("Jumlah kolom input:", len(X_cols_reg))
        st.write(X_cols_reg)

    sub_manual_r, sub_batch_r = st.tabs(["🧾 Input Manual (1 transaksi)", "📦 Prediksi Massal (CSV)"])

    # ---- Manual
    with sub_manual_r:
        st.markdown("Isi form ini untuk demo prediksi **Sales** untuk 1 transaksi.")

        default_od = pd.to_datetime(df_raw["Order Date"], errors="coerce").dropna()
        default_sd = pd.to_datetime(df_raw["Ship Date"], errors="coerce").dropna()

        with st.form("form_reg_manual"):
            colA, colB = st.columns(2)
            od = colA.date_input("Order Date (Regresi)", value=default_od.iloc[0].date() if len(default_od) else pd.Timestamp.today().date())
            sd = colB.date_input("Ship Date (Regresi)", value=default_sd.iloc[0].date() if len(default_sd) else pd.Timestamp.today().date())

            row_dict = build_manual_row_inputs(X_cols_reg, df_enriched, od, sd)
            submitted = st.form_submit_button("Prediksi Sales", type="primary")

        if submitted:
            try:
                X_one = pd.DataFrame([row_dict])
                pred = float(model_reg.predict(X_one)[0])
                st.success(f"✅ Predicted {target_reg}: **{pred:.2f}**")
                st.dataframe(X_one, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Gagal prediksi manual: {e}")

    # ---- Batch
    with sub_batch_r:
        st.markdown("Prediksi **nilai Sales** untuk seluruh baris pada CSV.")
        if st.button("Prediksi Sales (Massal)", type="primary"):
            try:
                X_pred = prepare_features(df_enriched, drop_cols_reg, target_reg)
                preds = model_reg.predict(X_pred)

                out = df_raw.copy()
                out[f"Predicted {target_reg}"] = preds

                t1, t2, t3 = st.tabs(["📌 Ringkasan", "📄 Tabel", "⬇️ Download"])

                with t1:
                    st.subheader(f"Ringkasan Predicted {target_reg}")
                    st.write(out[f"Predicted {target_reg}"].describe())

                with t2:
                    st.subheader("Preview hasil (50 baris)")
                    st.dataframe(out.head(50), use_container_width=True)

                with t3:
                    st.download_button(
                        f"Download hasil {target_reg} (.csv)",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name=f"superstore_predicted_{target_reg.lower()}.csv",
                        mime="text/csv"
                    )

                st.success("✅ Prediksi massal selesai!")
            except Exception as e:
                st.error(f"❌ Error prediksi massal: {e}")
