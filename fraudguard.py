"""
FraudGuard — Credit Card Fraud Detection Operations Center
Unified Single-Page Dashboard with Left-Pane Uploads and High-Performance Analytics.
Powered by xgboost_fraud_detector_deployment.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, io, base64, datetime, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI — Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;600;700;900&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #020813 !important;
    color: #e2e8f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"] { display: none; }

section[data-testid="stSidebar"] {
    background-color: #050811 !important;
    border-right: 1px solid #111c30 !important;
    padding-top: 10px !important;
}

.kpi-card-custom {
    background:#080f1e; border:1px solid #111c30; border-radius:12px;
    padding:16px 20px 12px 20px; position:relative; overflow:hidden;
    box-shadow:0 4px 12px rgba(0,0,0,0.25);
    transition:transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card-custom:hover { transform:translateY(-2px); border-color:#1a2f4c; }

.panel-container {
    background:#080f1e; border:1px solid #111c30; border-radius:12px;
    padding:20px; margin-bottom:20px; box-shadow:0 4px 10px rgba(0,0,0,0.2);
}
.panel-header-custom {
    font-size:10.5px; font-weight:700; text-transform:uppercase;
    letter-spacing:1.5px; color:#64748b; margin-bottom:15px;
}

div[data-testid="stRadio"] > div { flex-direction:row !important; gap:8px !important; }
div[data-testid="stRadio"] label {
    background:#080f1e !important; border:1px solid #111c30 !important;
    padding:6px 14px !important; border-radius:6px !important;
    color:#94a3b8 !important; font-size:12px !important; font-weight:600 !important;
}
div[data-testid="stRadio"] label[data-checked="true"] {
    background:#7c3aed !important; color:#fff !important; border-color:#7c3aed !important;
}
[data-baseweb="select"] > div {
    background:#080f1e !important; border-color:#111c30 !important;
    color:#e2e8f0 !important; border-radius:8px !important;
}

.custom-table-container { overflow-x:auto; }
.custom-table { width:100%; border-collapse:collapse; font-size:13px; color:#e2e8f0; }
.custom-table th {
    color:#64748b; font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:1px; border-bottom:1px solid #111c30; padding:12px 10px; text-align:left;
}
.custom-table td { padding:11px 10px; border-bottom:1px solid #0d172a; }
.custom-table tr:hover { background:rgba(255,255,255,0.02); }

.badge-status-p { padding:3px 9px; border-radius:6px; font-size:10px; font-weight:700; text-transform:uppercase; }
.badge-fraud-p  { background:rgba(255,45,85,.12);  color:#ff2d55; border:1px solid rgba(255,45,85,.25);  }
.badge-normal-p { background:rgba(16,185,129,.12); color:#10b981; border:1px solid rgba(16,185,129,.25); }
.badge-med-p    { background:rgba(255,184,0,.12);  color:#ffb800; border:1px solid rgba(255,184,0,.25);  }

div[data-testid="stFileUploader"] section {
    background:rgba(8,15,30,.6) !important;
    border:1px dashed rgba(124,58,237,.35) !important;
    border-radius:8px !important; padding:10px !important;
}
div[data-testid="stFileUploader"] section:hover {
    border-color:rgba(124,58,237,.7) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────
AMOUNT_MEDIAN, AMOUNT_IQR = 22.0,    71.565
TIME_MEDIAN,   TIME_IQR   = 84692.0, 85119.0
BASE_DATE = datetime.datetime(2025, 1, 1)

def scale_amount(a): return (np.array(a, dtype=float) - AMOUNT_MEDIAN) / AMOUNT_IQR
def scale_time(t):   return (np.array(t,  dtype=float) - TIME_MEDIAN)  / TIME_IQR
def fmt_cur(a): return f"${float(a):,.2f}"
def fmt_dt(s):  return (BASE_DATE + datetime.timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M")
def fmt_tm(s):  return (BASE_DATE + datetime.timedelta(seconds=float(s))).strftime("%I:%M %p")

def risk_label(score, hi, lo):
    if score >= hi: return "HIGH",   "#ff2d55", "badge-fraud-p"
    if score >= lo: return "MEDIUM", "#ffb800", "badge-med-p"
    return "LOW", "#10b981", "badge-normal-p"

def sparkline_b64(data, color):
    fig, ax = plt.subplots(figsize=(2.5, 0.45), dpi=100)
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    ax.plot(data, color=color, lw=1.6); ax.axis("off")
    fig.subplots_adjust(0,0,1,1)
    buf = io.BytesIO(); fig.savefig(buf, format="png", transparent=True, dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

PCA_COLS  = [f"V{i}" for i in range(1, 29)]
FEAT_COLS = PCA_COLS + ["scaled_Amount", "scaled_Time"]

# ─────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path = "xgboost_fraud_detector_deployment.pkl"
    if not os.path.exists(path):
        return None, [], ""
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    return pkg["model"], pkg.get("feature_columns", FEAT_COLS), pkg.get("reason", "")

model, feat_cols_pkg, deploy_reason = load_model()
model_ok = model is not None

# ─────────────────────────────────────────────────────────────────
#  SCORING FUNCTION (FAST BATCH VECTORISED)
# ─────────────────────────────────────────────────────────────────
def score_df(df):
    df = df.copy()
    df["scaled_Amount"] = scale_amount(df["Amount"].values)
    df["scaled_Time"]   = scale_time(df["Time"].values)
    cols = feat_cols_pkg if feat_cols_pkg else FEAT_COLS
    if model_ok:
        df["Risk_Score"] = model.predict_proba(df[cols])[:, 1]
    else:
        np.random.seed(42)
        df["Risk_Score"] = np.where(
            df.get("Class", 0) == 1,
            np.random.uniform(0.65, 0.99, len(df)),
            np.random.uniform(0.01, 0.28, len(df)),
        )
    return df

def add_anomaly_dist(df, centroid, max_dist):
    dists = np.sqrt(((df[PCA_COLS].values - centroid) ** 2).sum(axis=1))
    df["Anomaly_Dist"]      = dists
    df["Anomaly_Dist_Norm"] = dists / max_dist if max_dist > 0 else dists
    return df

# ─────────────────────────────────────────────────────────────────
#  PRESET TEST DATA
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def make_preset():
    np.random.seed(42)
    n_normal, n_fraud = 42, 8
    rows = []
    # Normal cases
    for i in range(n_normal):
        r = {"Transaction_ID": f"TXN-{10000+i}",
             "Time":  float(np.random.uniform(0, 172800)),
             "Amount": float(np.random.uniform(1.5, 2500)),
             "Class": 0}
        for j, (mu, sd) in enumerate(zip(
            [-0.01, 0.00, 0.01, 0.00, -0.01, 0.00, 0.00, 0.01,
              0.00, 0.01,  0.01, 0.00, -0.01, -0.00, -0.01, 0.01,
             -0.00, 0.00, 0.00,  0.00,  0.00,  0.00, -0.00, -0.00,
              0.00, 0.00, -0.00, -0.00],
            [1.96,1.65,1.52,1.42,1.38,1.33,1.24,1.19,
             1.15,1.10,1.02,0.99,0.96,0.95,0.92,0.88,
             0.86,0.84,0.81,0.77,0.73,0.72,0.62,0.61,
             0.52,0.48,0.40,0.33]
        ), 1):
            r[f"V{j}"] = float(np.random.normal(mu, sd))
        rows.append(r)

    # Fraud cases
    fraud_amounts = [1.79, 3.20, 0.77, 199.99, 4.50, 2.90, 8.99, 1.00]
    for i in range(n_fraud):
        r = {"Transaction_ID": f"TXN-FRAUD-{i+1}",
             "Time":  float(np.random.uniform(0, 172800)),
             "Amount": fraud_amounts[i],
             "Class": 1}
        for j, (mu, sd) in enumerate(zip(
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00, 0.00, 0.00, 0.00,-4.50, 0.00, 0.00,
            -3.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00, 0.00, 0.00],
            [1.5,1.3,1.2,1.1,1.1,1.0,1.0,0.9,
             0.9,0.9,0.8,0.8,0.8,0.7,0.7,0.7,
             0.7,0.6,0.6,0.6,0.6,0.5,0.5,0.5,
             0.4,0.4,0.3,0.3]
        ), 1):
            r[f"V{j}"] = float(np.random.normal(mu, sd))
        rows.append(r)

    df = pd.DataFrame(rows).sample(frac=1, random_state=99).reset_index(drop=True)
    df = score_df(df)
    centroid = df[df["Class"] == 0][PCA_COLS].mean().values
    max_dist = np.sqrt(((df[PCA_COLS].values - centroid)**2).sum(axis=1)).max()
    df = add_anomaly_dist(df, centroid, max_dist)
    return df, centroid, max_dist

preset_df, centroid_base, max_dist_base = make_preset()

# ─────────────────────────────────────────────────────────────────
#  PROCESS CSV UPLOAD (FAST)
# ─────────────────────────────────────────────────────────────────
def process_upload(raw):
    df = raw.copy()
    missing = [c for c in PCA_COLS + ["Amount","Time"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if "Transaction_ID" not in df.columns:
        df["Transaction_ID"] = [f"TXN-{100000+i}" for i in range(len(df))]
    if "Class" not in df.columns:
        df["Class"] = 0
    df = score_df(df)
    centroid = df[df["Class"]==0][PCA_COLS].mean().values if (df["Class"]==0).any() else df[PCA_COLS].mean().values
    max_d = np.sqrt(((df[PCA_COLS].values - centroid)**2).sum(axis=1)).max()
    df = add_anomaly_dist(df, centroid, max_d)
    return df

# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "active_df" not in st.session_state:
    st.session_state.active_df        = preset_df.copy()
    st.session_state.selected_txn     = preset_df["Transaction_ID"].iloc[0]
    st.session_state.data_source      = "Preset Test Data"
    st.session_state.thresh_high      = 0.65
    st.session_state.thresh_low       = 0.33
    st.session_state.overrides        = {}
    st.session_state.last_uploaded_fn = None

# Active dataframe with thresholds applied
def get_active():
    df = st.session_state.active_df.copy()
    hi = st.session_state.thresh_high
    lo = st.session_state.thresh_low
    s  = df["Risk_Score"]

    status = np.where(s < lo,  "Auto-Approved",
             np.where(s >= hi, "Auto-Declined", "Pending Review"))
    df["Status"]    = status
    df["Risk_Tier"] = df["Risk_Score"].apply(lambda val: risk_label(val, hi, lo)[0])
    df["Risk_Color"]= df["Risk_Score"].apply(lambda val: risk_label(val, hi, lo)[1])
    df["Risk_Badge"]= df["Risk_Score"].apply(lambda val: risk_label(val, hi, lo)[2])

    if st.session_state.overrides:
        ov = df["Transaction_ID"].map(st.session_state.overrides)
        has = ov.notna()
        df.loc[has, "Status"] = ov[has]
    return df

adf = get_active()
hi  = st.session_state.thresh_high
lo  = st.session_state.thresh_low


# ═════════════════════════════════════════════════════════════════
#  LEFT PANE (SIDEBAR) — FILE UPLOADER & CONTROLS
# ═════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:18px;padding-top:10px;">
      <svg viewBox="0 0 24 24" width="50" height="50"
           style="fill:none;stroke:#8b5cf6;stroke-width:1.5;
                  filter:drop-shadow(0 0 8px rgba(139,92,246,.45));">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <rect x="7" y="10" width="10" height="5" rx="1" style="fill:#8b5cf6;stroke:none;"/>
      </svg>
      <div style="font-size:16px;font-weight:800;color:#fff;letter-spacing:.5px;margin-top:6px;">
        Fraud Detection
        <span style="background:linear-gradient(90deg,#38bdf8,#8b5cf6);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     font-weight:900;">AI</span>
      </div>
      <div style="font-size:11px;color:#475569;margin-top:2px;">XGBoost · Real-time Scoring</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#111c30;margin-bottom:14px;'></div>",
                unsafe_allow_html=True)

    # ── UPLOAD SECTION ───────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#8b5cf6;text-transform:uppercase;
                letter-spacing:1.5px;margin-bottom:8px;">
      📁 DATA UPLOAD SECTION
    </div>""", unsafe_allow_html=True)

    up_file = st.file_uploader(
        "Upload PCA CSV File", type=["csv"],
        key="left_pane_uploader", label_visibility="collapsed"
    )
    st.markdown("""
    <div style="font-size:10.5px;color:#475569;margin-top:4px;line-height:1.5;">
      Format: CSV containing <code style="color:#38bdf8;">V1–V28</code>,
      <code style="color:#38bdf8;">Amount</code>, <code style="color:#38bdf8;">Time</code>
    </div>""", unsafe_allow_html=True)

    if up_file is not None:
        if st.session_state.last_uploaded_fn != up_file.name:
            with st.spinner("Scoring uploaded transactions…"):
                try:
                    raw_df = pd.read_csv(up_file)
                    proc_df = process_upload(raw_df)
                    st.session_state.active_df        = proc_df
                    st.session_state.selected_txn     = proc_df["Transaction_ID"].iloc[0]
                    st.session_state.data_source      = up_file.name
                    st.session_state.last_uploaded_fn = up_file.name
                    st.toast(f"✅ Scored {len(proc_df):,} transactions from {up_file.name}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"⚠️ {ex}")

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # Active dataset status card
    src_color = "#10b981" if st.session_state.data_source == "Preset Test Data" else "#a855f7"
    st.markdown(f"""
    <div style="background:#0d172a;border:1px solid #111c30;border-radius:8px;
                padding:10px 12px;margin-bottom:12px;">
      <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;
                  letter-spacing:1px;">Active Dataset</div>
      <div style="font-size:12.5px;font-weight:700;color:{src_color};
                  margin-top:3px;word-break:break-all;">{st.session_state.data_source}</div>
      <div style="font-size:10.5px;color:#475569;margin-top:2px;">
        {len(adf):,} transactions · {int(adf['Class'].sum())} fraud cases
      </div>
    </div>""", unsafe_allow_html=True)

    # Dataset controls
    if st.button("📊 Reset to Preset Data", key="reset_preset_btn", use_container_width=True):
        st.session_state.active_df        = preset_df.copy()
        st.session_state.selected_txn     = preset_df["Transaction_ID"].iloc[0]
        st.session_state.data_source      = "Preset Test Data"
        st.session_state.last_uploaded_fn = None
        st.toast("ℹ️ Reset to default preset test data")
        st.rerun()

    st.markdown("<div style='height:1px;background:#111c30;margin:16px 0;'></div>",
                unsafe_allow_html=True)

    # ── RISK THRESHOLD SLIDERS ────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;
                letter-spacing:1.5px;margin-bottom:10px;">
      ⚙️ RISK THRESHOLDS
    </div>""", unsafe_allow_html=True)

    new_hi = st.slider("🔴 High Risk (Decline)", 0.40, 0.95, float(hi), 0.01, key="sl_hi")
    new_lo = st.slider("🟡 Medium Risk (Review)", 0.10, 0.60, float(lo), 0.01, key="sl_lo")

    if new_hi != hi or new_lo != lo:
        if new_lo < new_hi:
            st.session_state.thresh_high = new_hi
            st.session_state.thresh_low  = new_lo
            st.rerun()
        else:
            st.warning("Low threshold must be smaller than High threshold.")

    st.markdown("<div style='height:1px;background:#111c30;margin:16px 0;'></div>",
                unsafe_allow_html=True)

    # ── SYSTEM HEALTH PANEL ──────────────────────────────────────
    n_high_s = (adf["Risk_Score"] >= hi).sum()
    fdr = (
        (adf[(adf["Class"]==1) & (adf["Risk_Score"]>=hi)].shape[0] /
         max(adf[adf["Class"]==1].shape[0], 1)) * 100
        if adf["Class"].sum() > 0 else 0
    )
    st.markdown(f"""
    <div style="background:#080f1e;border:1px solid #111c30;border-radius:10px;padding:12px;">
      <div style="font-size:9.5px;font-weight:700;color:#64748b;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:10px;">Model System Status</div>
      <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;">
          <span style="color:#94a3b8;">XGBoost Model</span>
          <span style="color:#10b981;font-weight:700;">{"✓ Active" if model_ok else "✗ Missing"}</span>
        </div>
        <div style="background:#111c30;height:4px;border-radius:2px;overflow:hidden;">
          <div style="background:{"#10b981" if model_ok else "#ff2d55"};width:100%;height:100%;"></div>
        </div>
      </div>
      <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;">
          <span style="color:#94a3b8;">High Risk Alerts</span>
          <span style="color:#ff2d55;font-weight:700;">{n_high_s}</span>
        </div>
        <div style="background:#111c30;height:4px;border-radius:2px;overflow:hidden;">
          <div style="background:#ff2d55;width:{min(n_high_s/max(len(adf),1)*100*4,100):.0f}%;height:100%;"></div>
        </div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;">
          <span style="color:#94a3b8;">Fraud Detection Rate</span>
          <span style="color:#00ff88;font-weight:700;">{fdr:.1f}%</span>
        </div>
        <div style="background:#111c30;height:4px;border-radius:2px;overflow:hidden;">
          <div style="background:linear-gradient(90deg,#10b981,#059669);width:{fdr:.0f}%;height:100%;"></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD (UNIFIED OPERATIONS CENTER & ANALYTICS)
# ═════════════════════════════════════════════════════════════════

now = datetime.datetime.now()
h1, h2 = st.columns([5, 5])
with h1:
    st.markdown("""
    <h1 style="margin:0;font-size:26px;font-weight:800;color:#fff;">
      Fraud Detection
      <span style="background:linear-gradient(90deg,#38bdf8,#8b5cf6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-weight:900;">AI Operations Center</span>
    </h1>
    <p style="margin:4px 0 0;color:#64748b;font-size:13px;font-weight:500;">
      Real-Time Monitoring · Latent Space Analytics · High-Risk Review
    </p>""", unsafe_allow_html=True)
with h2:
    st.markdown(f"""
    <div style="display:flex;gap:18px;align-items:center;justify-content:flex-end;padding-top:6px;">
      <div style="text-align:right;">
        <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:1px;">ROC-AUC</div>
        <div style="font-size:16px;font-weight:800;color:#10b981;font-family:monospace;">0.9789</div>
      </div>
      <div style="width:1px;height:26px;background:#111c30;"></div>
      <div style="text-align:right;">
        <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Engine</div>
        <div style="font-size:13px;font-weight:700;color:#e2e8f0;">XGBoost v2.0</div>
      </div>
      <div style="width:1px;height:26px;background:#111c30;"></div>
      <div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.2);
                  border-radius:30px;padding:4px 10px;font-size:10px;font-weight:750;
                  color:#10b981;display:flex;align-items:center;gap:6px;">
        <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                      background:#10b981;box-shadow:0 0 6px #10b981;"></span>
        SYSTEM ONLINE
      </div>
      <div style="background:#080f1e;border:1px solid #111c30;border-radius:8px;
                  padding:4px 10px;font-size:10.5px;font-weight:700;color:#94a3b8;
                  font-family:monospace;">
        📅 {now.strftime("%b %d, %Y")} &nbsp;|&nbsp; ⏰ {now.strftime("%I:%M %p")}
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

# ── TOP KPI CARDS (5 COLUMNS WITH SPARKLINES) ─────────────────────
total_v   = len(adf)
n_fraud_v = int(adf["Class"].sum())
n_high_v  = int((adf["Risk_Score"] >= hi).sum())
avg_sc_v  = adf["Risk_Score"].mean()
avg_ad_v  = adf["Anomaly_Dist_Norm"].mean()

df_srt  = adf.sort_values("Time").reset_index(drop=True)
samp    = df_srt.iloc[::max(1, len(df_srt)//40)].head(40)
sp_tot  = sparkline_b64(samp["Amount"].rolling(5,min_periods=1).mean().values,       "#a855f7")
sp_frd  = sparkline_b64(samp["Class"].cumsum().values,                               "#ff2d55")
sp_alrt = sparkline_b64((samp["Risk_Score"]>=hi).cumsum().values,                   "#ffb800")
sp_sc   = sparkline_b64(samp["Risk_Score"].rolling(5,min_periods=1).mean().values,   "#3b82f6")
sp_ad   = sparkline_b64(samp["Anomaly_Dist_Norm"].rolling(5,min_periods=1).mean().values,"#14b8a6")

kpi_configs = [
    ("TOTAL TRANSACTIONS", f"{total_v:,}", f"{st.session_state.data_source}", sp_tot, "#a855f7",
     '<svg viewBox="0 0 24 24" width="20" height="20" style="fill:none;stroke:#a855f7;stroke-width:2;"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>'),
    ("FRAUD CASES", f"{n_fraud_v:,}", f"{n_fraud_v/total_v*100:.1f}% of dataset", sp_frd, "#ff2d55",
     '<svg viewBox="0 0 24 24" width="20" height="20" style="fill:none;stroke:#ff2d55;stroke-width:2;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'),
    ("HIGH-RISK ALERTS", f"{n_high_v:,}", f"Risk Score ≥ {hi:.2f}", sp_alrt, "#ffb800",
     '<svg viewBox="0 0 24 24" width="20" height="20" style="fill:none;stroke:#ffb800;stroke-width:2;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>'),
    ("AVG RISK SCORE", f"{avg_sc_v:.4f}", "Mean XGBoost output", sp_sc, "#3b82f6",
     '<svg viewBox="0 0 24 24" width="20" height="20" style="fill:none;stroke:#3b82f6;stroke-width:2;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 11l2 2 4-4"/></svg>'),
    ("AVG ANOMALY DIST", f"{avg_ad_v:.4f}", "Normalised centroid dist", sp_ad, "#14b8a6",
     '<svg viewBox="0 0 24 24" width="20" height="20" style="fill:none;stroke:#14b8a6;stroke-width:2;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'),
]

kpi_cols = st.columns(5)
for col, (lbl, val, sub, spark, color, icon) in zip(kpi_cols, kpi_configs):
    with col:
        st.markdown(f"""
        <div class="kpi-card-custom">
          <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
            <div style="width:40px;height:40px;border-radius:50%;background:{color}15;
                        display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              {icon}</div>
            <div>
              <div style="font-size:9.5px;font-weight:700;color:#64748b;
                          letter-spacing:1.5px;text-transform:uppercase;">{lbl}</div>
              <div style="font-size:21px;font-weight:800;color:#fff;
                          margin-top:1px;font-family:monospace;">{val}</div>
              <div style="font-size:10.5px;color:#475569;margin-top:1px;">{sub}</div>
            </div>
          </div>
          <div style="margin-top:10px;margin-left:-20px;margin-right:-20px;margin-bottom:-13px;">
            <img src="data:image/png;base64,{spark}"
                 style="width:100%;height:32px;object-fit:cover;display:block;opacity:.8;">
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)

# ── SECTION 1: PCA LATENT SPACE, RISK DONUT & GAUGE (3 COLS) ─────
m1, m2, m3 = st.columns([5, 3.2, 3.8])

with m1:
    st.markdown('<div class="panel-container"><div class="panel-header-custom">PCA LATENT SPACE — PC1 vs PC2</div>', unsafe_allow_html=True)
    normal_s = adf[adf["Class"]==0].sample(min(600, (adf["Class"]==0).sum()), random_state=1)
    fraud_s  = adf[adf["Class"]==1].sample(min(400, (adf["Class"]==1).sum()), random_state=1)
    fig, ax = plt.subplots(figsize=(6.2, 3.65))
    fig.patch.set_facecolor("#080f1e"); ax.set_facecolor("#080f1e")
    ax.grid(True, color="#111c30", lw=0.5)
    for sp in ax.spines.values(): sp.set_color("#111c30")
    ax.scatter(normal_s["V1"], normal_s["V2"], c="#00d4ff", alpha=0.22, s=8, zorder=2)
    ax.scatter(fraud_s["V1"],  fraud_s["V2"],  c="#ff2d55", alpha=0.55, s=12, zorder=3)
    sel = st.session_state.selected_txn
    if sel in adf["Transaction_ID"].values:
        sr = adf[adf["Transaction_ID"]==sel].iloc[0]
        sx, sy = float(sr["V1"]), float(sr["V2"])
        ax.scatter(sx, sy, c="none", s=220, edgecolors="#00ff88", linewidths=1.0, alpha=0.4, zorder=4)
        ax.scatter(sx, sy, c="none", s=110, edgecolors="#00ff88", linewidths=1.8, alpha=0.8, zorder=4)
        ax.scatter(sx, sy, c="#fff", marker="*", s=80, edgecolors="#00ff88", linewidths=0.5, zorder=5)
    ax.set_xlabel("PC1 (V1)", color="#64748b", fontsize=7.5, labelpad=4)
    ax.set_ylabel("PC2 (V2)", color="#64748b", fontsize=7.5, labelpad=4)
    ax.tick_params(colors="#64748b", labelsize=7.5)
    handles = [
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#00d4ff",markersize=8,label=f"Normal ({len(normal_s)})"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#ff2d55",markersize=8,label=f"Fraud ({len(fraud_s)})"),
        Line2D([0],[0],marker="*",color="none",markerfacecolor="#fff",markeredgecolor="#00ff88",markersize=12,label="Selected"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", framealpha=0.1,
                    facecolor="#080f1e", edgecolor="#111c30", fontsize=8)
    for t in leg.get_texts(): t.set_color("#94a3b8")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95)
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown("""
    <div style="font-size: 11px; color: #64748b; margin-top: 10px; line-height: 1.45; text-align: left; border-top: 1px solid #111c30; padding-top: 8px;">
        <strong>Interpretation:</strong> This plot projects 28-dimensional PCA transaction features onto a 2D space. Legitimate transactions (blue) cluster tightly around the normal centroid, while anomalous/fraudulent transactions (red) are scattered outliers. The inspected transaction is highlighted with a green star (★).
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with m2:
    st.markdown('<div class="panel-container"><div class="panel-header-custom">RISK SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
    low_c  = int((adf["Risk_Score"] < lo).sum())
    med_c  = int(((adf["Risk_Score"] >= lo) & (adf["Risk_Score"] < hi)).sum())
    high_c = int((adf["Risk_Score"] >= hi).sum())

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#080f1e"); ax.set_facecolor("#080f1e")
    if total_v > 0:
        ax.pie([max(low_c,1), max(med_c,1), max(high_c,1)],
               colors=["#3b82f6","#ffb800","#ff2d55"],
               startangle=90, counterclock=False,
               wedgeprops=dict(width=0.25, edgecolor="#080f1e", linewidth=2.5))
    ax.text(0, 0.08, f"{total_v:,}", ha="center", va="center",
            color="#fff", fontsize=22, fontweight="bold", fontfamily="monospace")
    ax.text(0, -0.15, "Total", ha="center", va="center",
            color="#64748b", fontsize=10.5, fontweight="bold")
    ax.axis("equal")
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.markdown(f"""
    <div style="margin-top:10px;font-size:11.5px;border-top:1px solid #111c30;
                padding-top:12px;display:flex;flex-direction:column;gap:6px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#3b82f6;"></span>
          Low Risk (&lt; {lo:.2f})
        </span>
        <span style="font-weight:700;font-family:monospace;">{low_c/total_v*100:.1f}%&nbsp;<span style="color:#64748b;font-weight:400;">{low_c:,}</span></span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ffb800;"></span>
          Medium ({lo:.2f} – {hi:.2f})
        </span>
        <span style="font-weight:700;font-family:monospace;">{med_c/total_v*100:.1f}%&nbsp;<span style="color:#64748b;font-weight:400;">{med_c:,}</span></span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff2d55;"></span>
          High Risk (≥ {hi:.2f})
        </span>
        <span style="font-weight:700;font-family:monospace;">{high_c/total_v*100:.1f}%&nbsp;<span style="color:#64748b;font-weight:400;">{high_c:,}</span></span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with m3:
    st.markdown('<div class="panel-container"><div class="panel-header-custom">FRAUD SCORE GAUGE &amp; INSPECTION</div>', unsafe_allow_html=True)
    sel_tier = st.radio("Tier:", ["All","High","Medium","Low"], horizontal=True, key="dash_tier")
    if sel_tier == "All":    pool = adf.sort_values("Risk_Score", ascending=False).head(500)
    elif sel_tier == "High": pool = adf[adf["Risk_Score"] >= hi].sort_values("Risk_Score", ascending=False).head(500)
    elif sel_tier == "Medium": pool = adf[(adf["Risk_Score"]>=lo)&(adf["Risk_Score"]<hi)].sort_values("Risk_Score",ascending=False).head(500)
    else: pool = adf[adf["Risk_Score"] < lo].sort_values("Risk_Score", ascending=False).head(500)

    txn_list = pool["Transaction_ID"].tolist()
    if not txn_list:
        st.info("No transactions in this tier.")
    else:
        if st.session_state.selected_txn not in txn_list:
            st.session_state.selected_txn = txn_list[0]
        chosen = st.selectbox(f"Select Transaction ({len(txn_list)})", txn_list,
                              index=txn_list.index(st.session_state.selected_txn),
                              key="dash_sel")
        st.session_state.selected_txn = chosen
        rec = adf[adf["Transaction_ID"]==chosen].iloc[0]
        curr_score = float(rec["Risk_Score"])

        gc1, gc2 = st.columns([6, 4])
        with gc1:
            fig_g, ax_g = plt.subplots(figsize=(4.0, 2.7))
            fig_g.patch.set_facecolor("#080f1e"); ax_g.set_facecolor("#080f1e")
            ax_g.set_xlim(-1.1,1.1); ax_g.set_ylim(-0.45,1.1)
            ax_g.set_aspect("equal"); ax_g.axis("off")
            cx, cy, Ro, Ri = 0.0, 0.0, 1.0, 0.65
            W = Ro - Ri
            ax_g.add_patch(Wedge((cx,cy), Ro, 120.6, 180, width=W, facecolor="#10b981", alpha=0.95, edgecolor="none"))
            ax_g.add_patch(Wedge((cx,cy), Ro, 57.8, 120.6, width=W, facecolor="#ffb800", alpha=0.95, edgecolor="none"))
            ax_g.add_patch(Wedge((cx,cy), Ro, 0.0, 57.8, width=W, facecolor="#ff2d55", alpha=0.95, edgecolor="none"))
            ax_g.add_patch(Wedge((cx,cy), Ro, 180, 360, width=W, facecolor="#111c30", alpha=0.6, edgecolor="none"))

            needle_angle = 180 - curr_score * 180
            nx = Ri * np.cos(np.radians(needle_angle))
            ny = Ri * np.sin(np.radians(needle_angle))
            ax_g.plot([cx, nx*1.1], [cy, ny*1.1], color="#fff", lw=2.5, solid_capstyle="round", zorder=10)
            ax_g.scatter(cx, cy, color="#fff", s=28, zorder=11)
            score_col = "#ff2d55" if curr_score>=hi else ("#ffb800" if curr_score>=lo else "#10b981")
            ax_g.text(0, -0.25, f"{curr_score:.4f}", ha="center", va="center",
                      color=score_col, fontsize=17, fontweight="bold", fontfamily="monospace")
            ax_g.text(0, -0.38, "Risk Score", ha="center", va="center", color="#64748b", fontsize=9)
            fig_g.subplots_adjust(0.02, 0.02, 0.98, 0.98)
            st.pyplot(fig_g, use_container_width=True); plt.close(fig_g)

        with gc2:
            lbl_g, clr_g, _ = risk_label(curr_score, hi, lo)
            st.markdown(f"""
            <div style="padding-top:6px;display:flex;flex-direction:column;gap:8px;">
              <div style="background:{clr_g}15;border:1px solid {clr_g}30;
                          border-radius:8px;padding:8px;text-align:center;">
                <div style="font-size:8.5px;color:#64748b;font-weight:700;
                            text-transform:uppercase;letter-spacing:1px;">Risk Tier</div>
                <div style="font-size:18px;font-weight:900;color:{clr_g};margin-top:2px;">{lbl_g}</div>
              </div>
              <div style="background:#0d172a;border-radius:8px;padding:8px;text-align:center;">
                <div style="font-size:8.5px;color:#64748b;font-weight:700;
                            text-transform:uppercase;letter-spacing:1px;">Amount</div>
                <div style="font-size:14px;font-weight:700;color:#fff;margin-top:2px;">{fmt_cur(rec['Amount'])}</div>
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── SECTION 2: ALERTS REVIEW QUEUE & ACTION PANEL ──────────────
b1, b2 = st.columns([6.5, 3.5])
with b1:
    st.markdown('<div class="panel-container"><div class="panel-header-custom">RECENT HIGH-RISK ALERTS (REVIEW QUEUE)</div>', unsafe_allow_html=True)
    alerts_queue = adf[adf["Risk_Score"]>=hi].sort_values("Risk_Score", ascending=False).head(5)
    if len(alerts_queue)==0:
        st.success("✅ No high-risk alerts. All transactions safe.")
    else:
        rows_h = ""
        for _, r in alerts_queue.iterrows():
            rows_h += f"""<tr>
              <td style="width:12px;padding-right:0;">
                <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                             background:#ff2d55;box-shadow:0 0 6px #ff2d55;"></span>
              </td>
              <td style="font-weight:700;color:#fff;font-family:monospace;font-size:12px;">{r['Transaction_ID']}</td>
              <td style="font-weight:600;">{fmt_cur(r['Amount'])}</td>
              <td style="font-weight:600;color:#94a3b8;">{fmt_tm(r['Time'])}</td>
              <td style="font-weight:800;color:#ff2d55;font-family:monospace;">{float(r['Risk_Score']):.4f}</td>
              <td style="color:#64748b;font-family:monospace;">{float(r['Anomaly_Dist_Norm']):.3f}</td>
              <td><span class="badge-status-p badge-fraud-p">{r['Status']}</span></td>
            </tr>"""
        st.markdown(f"""<div class="custom-table-container"><table class="custom-table">
          <thead><tr><th></th><th>Transaction ID</th><th>Amount</th><th>Time</th>
          <th>Risk Score</th><th>Anomaly Dist</th><th>Status</th></tr></thead>
          <tbody>{rows_h}</tbody></table></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b2:
    st.markdown('<div class="panel-container"><div class="panel-header-custom">ANALYST ACTION PANEL</div>', unsafe_allow_html=True)
    if st.session_state.selected_txn in adf["Transaction_ID"].values:
        rec_act = adf[adf["Transaction_ID"]==st.session_state.selected_txn].iloc[0]
        st.markdown(f"""
        <div style="background:#0d172a;border-radius:8px;padding:12px;margin-bottom:12px;">
          <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase;">Selected Transaction</div>
          <div style="font-size:15px;font-weight:800;color:#fff;font-family:monospace;margin-top:2px;">
            {st.session_state.selected_txn}
          </div>
          <div style="font-size:11px;color:#94a3b8;margin-top:3px;">
            Score: <span style="color:#ff2d55;font-weight:700;">{float(rec_act['Risk_Score']):.4f}</span> ·
            Amount: <span style="color:#fff;">{fmt_cur(rec_act['Amount'])}</span>
          </div>
        </div>""", unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        with ac1:
            if st.button("✅ Approve", key="act_approve", use_container_width=True):
                st.session_state.overrides[st.session_state.selected_txn] = "Approved"
                st.toast(f"✅ Approved {st.session_state.selected_txn}"); st.rerun()
        with ac2:
            if st.button("🚫 Decline", key="act_decline", use_container_width=True):
                st.session_state.overrides[st.session_state.selected_txn] = "Declined"
                st.toast(f"🚫 Declined {st.session_state.selected_txn}"); st.rerun()

        if st.session_state.selected_txn in st.session_state.overrides:
            st.info(f"Override: **{st.session_state.overrides[st.session_state.selected_txn]}**")
    st.markdown("</div>", unsafe_allow_html=True)







# ── SECTION 4: FULL SCORED TRANSACTIONS LOG ──────────────────────
st.markdown("""
<div style="font-size:13px;font-weight:800;color:#fff;text-transform:uppercase;
            letter-spacing:1.5px;margin:26px 0 16px;display:flex;align-items:center;gap:10px;">
  <span>💳 SCORED TRANSACTIONS LOG</span>
  <div style="flex-grow:1;height:1px;background:#111c30;"></div>
</div>""", unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns([2.5, 2.5, 2.5, 4.5])
with fc1:
    tier_f = st.radio("Risk Tier Filter:", ["All","HIGH","MEDIUM","LOW"], horizontal=True, key="log_tier")
with fc2:
    sort_f = st.radio("Sort Transactions:", ["Risk ↓","Amount ↓","Time ↑"], horizontal=True, key="log_sort")
with fc3:
    show_f = st.radio("Label Filter:", ["All","Fraud Only","Normal Only"], horizontal=True, key="log_show")

disp = adf.copy()
if tier_f == "HIGH":   disp = disp[disp["Risk_Score"] >= hi]
elif tier_f == "MEDIUM": disp = disp[(disp["Risk_Score"]>=lo)&(disp["Risk_Score"]<hi)]
elif tier_f == "LOW":  disp = disp[disp["Risk_Score"] < lo]

if show_f == "Fraud Only": disp = disp[disp["Class"]==1]
elif show_f == "Normal Only": disp = disp[disp["Class"]==0]

if sort_f == "Risk ↓": disp = disp.sort_values("Risk_Score", ascending=False)
elif sort_f == "Amount ↓": disp = disp.sort_values("Amount", ascending=False)
else: disp = disp.sort_values("Time")

# Render top 100 rows for sub-millisecond DOM table rendering
disp_view = disp.head(100)

st.markdown(f"""
<div class="panel-container">
<div class="panel-header-custom">
  SCORED LOG — Showing top {len(disp_view)} of {len(disp):,} matching transactions (full data in CSV download)
</div>""", unsafe_allow_html=True)

rows_html = ""
for _, r in disp_view.iterrows():
    sc   = float(r["Risk_Score"])
    ad   = float(r["Anomaly_Dist_Norm"])
    lbl, clr, badge = risk_label(sc, hi, lo)
    pred = "FRAUD" if sc >= 0.5 else "NORMAL"
    pb   = "badge-fraud-p" if pred=="FRAUD" else "badge-normal-p"
    true_lbl = "FRAUD" if int(r.get("Class",0))==1 else "NORMAL"
    tb   = "badge-fraud-p" if true_lbl=="FRAUD" else "badge-normal-p"
    bar_w = int(sc * 100)
    rows_html += f"""<tr style="background:{'rgba(255,45,85,.03)' if sc>=hi else 'transparent'};">
      <td style="font-family:monospace;color:#94a3b8;font-size:12px;">{r['Transaction_ID']}</td>
      <td style="font-weight:700;color:#fff;">{fmt_cur(r['Amount'])}</td>
      <td style="color:#64748b;">{fmt_dt(r['Time'])}</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:60px;background:#111c30;border-radius:3px;height:4px;">
            <div style="width:{bar_w}%;background:{clr};height:100%;border-radius:3px;"></div>
          </div>
          <span style="font-weight:800;color:{clr};font-family:monospace;font-size:12px;">{sc:.4f}</span>
        </div>
      </td>
      <td style="color:#64748b;font-family:monospace;font-size:12px;">{ad:.3f}</td>
      <td><span class="badge-status-p {badge}">{lbl}</span></td>
      <td><span class="badge-status-p {pb}">{pred}</span></td>
    </tr>"""

st.markdown(f"""
<div class="custom-table-container" style="max-height:500px;overflow-y:auto;">
  <table class="custom-table">
    <thead><tr>
      <th>Transaction ID</th><th>Amount</th><th>Time</th>
      <th>Risk Score</th><th>Anomaly Dist</th><th>Risk Tier</th>
      <th>Prediction</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>""", unsafe_allow_html=True)



st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 10px;font-size:11px;color:#1e293b;">
  FraudGuard AI · Credit Card Fraud Detection Operations Center · IoT Academy Capstone ·
  <span style="color:#334155;">XGBoost + Streamlit</span>
</div>""", unsafe_allow_html=True)
