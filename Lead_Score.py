import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
import warnings
import hashlib
import sqlite3
from datetime import datetime
import time
from io import BytesIO

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="LeadScore Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATABASE FUNCTIONS (unchanged)
# ============================================================================

def init_database():
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  email TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  last_login TIMESTAMP,
                  is_active BOOLEAN DEFAULT 1,
                  role TEXT DEFAULT 'user')''')
    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  action TEXT,
                  details TEXT,
                  leads_scored INTEGER,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  logout_time TIMESTAMP,
                  is_active BOOLEAN DEFAULT 1,
                  session_token TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    try:
        c.execute("PRAGMA table_info(sessions)")
        columns = [column[1] for column in c.fetchall()]
        if 'is_active' not in columns:
            c.execute("ALTER TABLE sessions ADD COLUMN is_active BOOLEAN DEFAULT 1")
            conn.commit()
        if 'session_token' not in columns:
            c.execute("ALTER TABLE sessions ADD COLUMN session_token TEXT")
            conn.commit()
    except Exception:
        pass
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)",
                  ('admin', admin_password, 'admin@leadscore.com', 'admin'))
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    try:
        password_hash = hash_password(password)
        c.execute("SELECT id, username, role, is_active FROM users WHERE username = ? AND password_hash = ?",
                  (username, password_hash))
        user = c.fetchone()
        if user and user[3]:
            c.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user[0]))
            session_token = hashlib.md5(f"{user[0]}{datetime.now()}".encode()).hexdigest()
            try:
                c.execute("UPDATE sessions SET is_active = 0, logout_time = ? WHERE user_id = ? AND is_active = 1",
                          (datetime.now(), user[0]))
            except sqlite3.OperationalError:
                pass
            try:
                c.execute("INSERT INTO sessions (user_id, login_time, is_active, session_token) VALUES (?, ?, ?, ?)",
                          (user[0], datetime.now(), 1, session_token))
            except sqlite3.OperationalError:
                c.execute("INSERT INTO sessions (user_id, login_time) VALUES (?, ?)",
                          (user[0], datetime.now()))
            conn.commit()
            conn.close()
            return {'id': user[0], 'username': user[1], 'role': user[2], 'is_active': user[3], 'session_token': session_token}
        conn.close()
        return None
    except Exception:
        conn.close()
        return None

def create_user_by_admin(username, password, email):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    try:
        password_hash = hash_password(password)
        c.execute("INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)",
                  (username, password_hash, email, 'user'))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def logout_user(user_id):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("UPDATE sessions SET is_active = 0, logout_time = ? WHERE user_id = ? AND is_active = 1",
                  (datetime.now(), user_id))
    except sqlite3.OperationalError:
        c.execute("UPDATE sessions SET logout_time = ? WHERE user_id = ? AND logout_time IS NULL",
                  (datetime.now(), user_id))
    conn.commit()
    conn.close()

def log_usage(user_id, action, details="", leads_scored=0):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO usage_logs (user_id, action, details, leads_scored) VALUES (?, ?, ?, ?)",
              (user_id, action, details, leads_scored))
    conn.commit()
    conn.close()

def get_user_stats(user_id):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM usage_logs WHERE user_id = ? AND action = 'score_leads'", (user_id,))
    total_scorings = c.fetchone()[0]
    c.execute("SELECT SUM(leads_scored) FROM usage_logs WHERE user_id = ? AND action = 'score_leads'", (user_id,))
    total_leads = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,))
    total_logins = c.fetchone()[0]
    conn.close()
    return {'total_scorings': total_scorings, 'total_leads': total_leads, 'total_logins': total_logins}

def get_all_users():
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, username, email, created_at, last_login, is_active, role FROM users ORDER BY created_at DESC")
    users = c.fetchall()
    conn.close()
    return users

def get_currently_logged_in_users():
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("""SELECT u.id, u.username, u.email, s.login_time, u.role
                     FROM sessions s JOIN users u ON s.user_id = u.id
                     WHERE s.is_active = 1 ORDER BY s.login_time DESC""")
        active_users = c.fetchall()
    except sqlite3.OperationalError:
        active_users = []
    conn.close()
    return active_users

def get_all_user_activities():
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""SELECT u.username, l.action, l.details, l.leads_scored, l.timestamp
                 FROM usage_logs l JOIN users u ON l.user_id = u.id
                 ORDER BY l.timestamp DESC LIMIT 100""")
    activities = c.fetchall()
    conn.close()
    return activities

def get_system_stats():
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
    total_users = c.fetchone()[0]
    try:
        c.execute("SELECT COUNT(*) FROM sessions WHERE is_active = 1")
        currently_online = c.fetchone()[0]
    except sqlite3.OperationalError:
        currently_online = 0
    c.execute("SELECT COUNT(*) FROM usage_logs WHERE action = 'score_leads'")
    total_scorings = c.fetchone()[0]
    c.execute("SELECT SUM(leads_scored) FROM usage_logs WHERE action = 'score_leads'")
    total_leads = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM sessions WHERE DATE(login_time) = DATE('now')")
    today_logins = c.fetchone()[0]
    conn.close()
    return {'total_users': total_users, 'currently_online': currently_online,
            'total_scorings': total_scorings, 'total_leads': total_leads, 'today_logins': today_logins}

def toggle_user_status(user_id, is_active):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("UPDATE users SET is_active = ? WHERE id = ?", (is_active, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_user_activity_details(user_id):
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""SELECT action, details, leads_scored, timestamp FROM usage_logs
                 WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20""", (user_id,))
    activities = c.fetchall()
    conn.close()
    return activities

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def map_probability_to_category(prob_score):
    if prob_score >= 70:
        return "Hot"
    elif prob_score >= 40:
        return "Warm"
    else:
        return "Cold"

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_resource
def train_model(df):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.markdown("🔧 **Step 1/5:** Feature Engineering...")
    progress_bar.progress(20)
    if "budget_min" in df.columns and "budget_max" in df.columns:
        df["budget_mid"] = df[["budget_min", "budget_max"]].mean(axis=1)
    elif "budget" in df.columns:
        df["budget_mid"] = pd.to_numeric(df["budget"], errors='coerce')
    else:
        df["budget_mid"] = np.nan
    if df["budget_mid"].notna().any():
        min_b, max_b = df["budget_mid"].min(), df["budget_mid"].max()
        if min_b == max_b or pd.isna(min_b) or pd.isna(max_b):
            df["budget_match"] = 1.0
        else:
            df["budget_match"] = (df["budget_mid"] - min_b) / (max_b - min_b)
    else:
        df["budget_match"] = 0.5
    if "preferred_area" in df.columns:
        area_freq = df["preferred_area"].fillna("unknown").value_counts(normalize=True)
        df["area_match"] = df["preferred_area"].fillna("unknown").map(area_freq).fillna(0.5)
    else:
        df["area_match"] = 0.5
    beh_cols = ["views_count", "avg_view_time_sec", "saved_properties", "repeated_visits"]
    for c in beh_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in beh_cols:
        mx = df[c].max()
        if mx > 0:
            df[c + "_norm"] = df[c] / mx
        else:
            df[c + "_norm"] = 0.0
    df["engagement_score"] = (0.4 * df["views_count_norm"] + 0.2 * df["avg_view_time_sec_norm"] +
                               0.25 * df["saved_properties_norm"] + 0.15 * df["repeated_visits_norm"])
    inter_cols = ["whatsapp_clicks", "call_clicks", "chat_messages"]
    for c in inter_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["total_interactions"] = df[inter_cols].sum(axis=1)
    if "last_active_time" in df.columns:
        df["last_active_time"] = pd.to_datetime(df["last_active_time"], errors="coerce")
        now = pd.Timestamp.now()
        df["days_since_active"] = (now - df["last_active_time"]).dt.days.fillna(999)
        df["recency_score"] = 1 / (1 + df["days_since_active"])
    else:
        df["recency_score"] = 0.0
    status_text.markdown("📊 **Step 2/5:** Preparing Features...")
    progress_bar.progress(40)
    feature_cols = ["budget_match", "area_match", "engagement_score", "total_interactions", "recency_score"]
    if "source" in df.columns:
        feature_cols.append("source")
    if "bhk" in df.columns:
        feature_cols.append("bhk")
    X = df[feature_cols].copy()
    y = None
    if "converted" in df.columns:
        y = pd.to_numeric(df["converted"], errors="coerce")
    if y is None or y.isna().all():
        status_text.markdown("🤖 **Using unsupervised learning:** Creating pseudo-labels with KMeans...")
        numeric_for_kmeans = X.select_dtypes(include=[np.number]).fillna(0)
        kmeans = KMeans(n_clusters=2, random_state=42)
        pseudo_labels = kmeans.fit_predict(numeric_for_kmeans)
        y = pd.Series(pseudo_labels, index=X.index)
    else:
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].astype(int).reset_index(drop=True)
    if len(X) < 10:
        raise ValueError("Not enough data to train model after cleaning")
    status_text.markdown("🔨 **Step 3/5:** Building ML Pipeline...")
    progress_bar.progress(60)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    transformers = []
    if num_cols:
        num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("num", num_transformer, num_cols))
    if cat_cols:
        cat_transformer = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", cat_transformer, cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
    pipeline = Pipeline([("preprocess", preprocessor), ("rf", rf)])
    status_text.markdown("🎯 **Step 4/5:** Training Model...")
    progress_bar.progress(80)
    stratify_y = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify_y)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = None
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            pass
    status_text.markdown("✨ **Step 5/5:** Scoring All Leads...")
    progress_bar.progress(100)
    df_scored = df.copy()
    lead_probability = pipeline.predict_proba(X)[:, 1]
    df_scored.loc[X.index, "lead_score"] = (lead_probability * 100).round(0).astype(int)
    df_scored["lead_score"] = df_scored["lead_score"].fillna(0).astype(int)
    df_scored["lead_category"] = df_scored["lead_score"].apply(map_probability_to_category)
    status_text.markdown("✅ **Model Training Complete!**")
    return pipeline, df_scored, feature_cols, accuracy, roc_auc

def create_donut_chart(value, title, color, bg_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 13, 'color': '#64748b', 'family': 'DM Sans'}},
        number={'font': {'size': 28, 'color': '#1e293b', 'family': 'DM Sans'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 0, 'visible': False},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': bg_color,
            'borderwidth': 0,
            'steps': [{'range': [0, 100], 'color': bg_color}],
        }
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'family': 'DM Sans'})
    return fig

# ============================================================================
# NEW THEME CSS — Navy Sidebar + White Card Dashboard
# ============================================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }

/* ─── App Background ─── */
[data-testid="stAppViewContainer"] {
    background-color: #f0f4f8 !important;
}
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ─── Sidebar: Deep Navy ─── */
[data-testid="stSidebar"] {
    background: #0f2044 !important;
    padding: 0 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebar"] * {
    color: #c8d8f0 !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small {
    color: #9ab0cc !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] h3 {
    color: #e8f0fc !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 0 !important;
    margin: 1rem 0 0.5rem 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #f97316 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: none !important;
    transition: background 0.2s !important;
    text-transform: none !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ea6c0a !important;
    transform: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e3660 !important;
    margin: 1rem 0 !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.4rem !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.75rem !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.1) !important;
}

/* ─── Top Page Header Bar ─── */
.page-topbar {
    background: white;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #e8edf2;
}
.page-topbar-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f2044;
    margin: 0;
}
.page-topbar-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0;
}
.topbar-badge {
    background: #f97316;
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.03em;
}

/* ─── Stat Cards (white, shadow) ─── */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 20px 20px 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #e8edf2;
    position: relative;
    overflow: hidden;
    min-height: 110px;
}
.stat-card-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    margin-bottom: 12px;
}
.stat-card-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f2044;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-card-label {
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stat-card-accent {
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 0 12px 12px 0;
}

/* ─── Section Cards ─── */
.section-card {
    background: white;
    border-radius: 12px;
    padding: 20px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #e8edf2;
    margin-bottom: 16px;
}
.section-card-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #0f2044;
    margin: 0 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.section-card-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 0 0 16px 0;
}

/* ─── Tabs: Clean pill style ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: white !important;
    padding: 6px !important;
    border-radius: 10px !important;
    border: 1px solid #e8edf2 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    backdrop-filter: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border: none !important;
    font-size: 0.85rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #f1f5f9 !important;
    color: #0f2044 !important;
}
.stTabs [aria-selected="true"] {
    background: #0f2044 !important;
    color: white !important;
    box-shadow: none !important;
}

/* ─── Buttons ─── */
.stButton > button {
    background: #0f2044 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: background 0.2s !important;
    box-shadow: none !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}
.stButton > button:hover {
    background: #1a3563 !important;
    transform: none !important;
    box-shadow: none !important;
}
.stDownloadButton > button {
    background: #059669 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: background 0.2s !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: #047857 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ─── DataFrames ─── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    border: 1px solid #e8edf2 !important;
}

/* ─── Metrics ─── */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #0f2044 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ─── Headings in main area ─── */
h1, h2, h3 {
    color: #0f2044 !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0 !important;
    margin-top: 0 !important;
}
h3 { font-size: 1rem !important; }

/* ─── Alerts ─── */
.stAlert {
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    border-left: 3px solid !important;
}

/* ─── Progress Bar ─── */
.stProgress > div > div > div > div {
    background: #f97316 !important;
    border-radius: 6px !important;
}

/* ─── Expanders ─── */
.streamlit-expanderHeader {
    background: #f8fafc !important;
    border-radius: 8px !important;
    border: 1px solid #e8edf2 !important;
    font-weight: 600 !important;
    color: #0f2044 !important;
    padding: 0.75rem 1rem !important;
    backdrop-filter: none !important;
}

/* ─── Inputs ─── */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #d1d9e6 !important;
    background: white !important;
    color: #0f2044 !important;
    padding: 0.55rem 0.75rem !important;
    font-size: 0.875rem !important;
    backdrop-filter: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0f2044 !important;
    box-shadow: 0 0 0 2px rgba(15,32,68,0.12) !important;
}

/* ─── Slider ─── */
.stSlider > div > div > div {
    background: #f97316 !important;
}

/* ─── Sidebar Nav Items ─── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 20px;
    border-radius: 8px;
    margin: 2px 12px;
    cursor: pointer;
    color: #9ab0cc;
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.15s;
}
.nav-item:hover { background: rgba(255,255,255,0.08); color: white; }
.nav-item.active { background: #f97316; color: white; }
.nav-icon { font-size: 16px; width: 20px; text-align: center; }

/* ─── User avatar in sidebar ─── */
.sidebar-profile {
    padding: 24px 20px 16px 20px;
    border-bottom: 1px solid #1e3660;
    margin-bottom: 8px;
}
.sidebar-avatar {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #f97316, #fb923c);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; color: white; font-weight: 700;
    margin-bottom: 10px;
}
.sidebar-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8f0fc !important;
    margin: 0 0 2px 0;
}
.sidebar-role {
    font-size: 0.73rem;
    color: #6b8aaa !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ─── Stat mini pills ─── */
.stat-pill {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    color: #fb923c;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 0;
}

/* ─── Category badges ─── */
.badge-hot { background:#fee2e2; color:#dc2626; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-warm { background:#fff7ed; color:#ea580c; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-cold { background:#eff6ff; color:#2563eb; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

/* ─── Online dot ─── */
.dot-online {
    display: inline-block;
    width: 8px; height: 8px;
    background: #10b981;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 6px #10b981;
}

/* ─── Mobile ─── */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .stat-card-value { font-size: 1.4rem !important; }
}
</style>
"""

# ============================================================================
# LOGIN PAGE — new clean theme
# ============================================================================

def show_login_page():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
    * { font-family: 'DM Sans', sans-serif !important; }
    [data-testid="stAppViewContainer"] {
        background: #0f2044 !important;
    }
    .login-wrap {
        max-width: 420px;
        margin: 60px auto 0 auto;
        background: white;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 24px 64px rgba(0,0,0,0.4);
    }
    .login-top {
        background: #0f2044;
        padding: 32px 28px 24px 28px;
        text-align: center;
        border-bottom: 3px solid #f97316;
    }
    .login-logo {
        width: 56px; height: 56px;
        background: #f97316;
        border-radius: 14px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 28px;
        margin-bottom: 14px;
    }
    .login-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin: 0 0 4px 0;
    }
    .login-sub {
        font-size: 0.82rem;
        color: #6b8aaa;
        margin: 0;
    }
    .login-body {
        padding: 28px 28px 24px 28px;
    }
    .login-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 6px;
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTextInput > div > div > input {
        border: 1.5px solid #d1d9e6 !important;
        border-radius: 8px !important;
        background: #f8fafc !important;
        color: #0f2044 !important;
        padding: 10px 14px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #f97316 !important;
        box-shadow: 0 0 0 3px rgba(249,115,22,0.15) !important;
    }
    .stButton > button {
        background: #f97316 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        width: 100% !important;
        box-shadow: none !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: #ea6c0a !important;
        transform: none !important;
    }
    .feat-item {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.82rem; color: #64748b;
        padding: 4px 0;
    }
    .feat-dot { width: 6px; height: 6px; background: #f97316; border-radius: 50%; flex-shrink: 0; }
    .login-footer {
        text-align: center; padding: 0 28px 24px 28px;
        font-size: 0.75rem; color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown("""
        <div class="login-wrap">
          <div class="login-top">
            <div class="login-logo">🎯</div>
            <h1 class="login-title">LeadScore Pro</h1>
            <p class="login-sub">AI-Powered Lead Intelligence Platform</p>
          </div>
          <div class="login-body">
        """, unsafe_allow_html=True)

        st.markdown('<label class="login-label">Username</label>', unsafe_allow_html=True)
        username = st.text_input("", key="login_username", placeholder="Enter username", label_visibility="collapsed")
        st.markdown('<label class="login-label" style="margin-top:14px;">Password</label>', unsafe_allow_html=True)
        password = st.text_input("", type="password", key="login_password", placeholder="Enter password", label_visibility="collapsed")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Sign In →", use_container_width=True, key="login_btn"):
                if username and password:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        log_usage(user['id'], 'login')
                        st.success(f"Welcome, {username}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                else:
                    st.warning("Please fill in both fields.")
        with col_b:
          if False:  # hidden button
               if st.button("Demo Creds", use_container_width=True, key="demo_btn"):
                  with st.expander("Credentials", expanded=True):
                    st.code("Username: admin\nPassword: admin123")

        st.markdown("""
          </div>
          <div style="padding: 0 28px 20px 28px; border-top: 1px solid #f1f5f9; margin-top: 4px;">
            <p style="font-size:0.75rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin: 14px 0 8px 0;">Platform Features</p>
            <div class="feat-item"><span class="feat-dot"></span>AI-powered scoring engine</div>
            <div class="feat-item"><span class="feat-dot"></span>Real-time analytics dashboard</div>
            <div class="feat-item"><span class="feat-dot"></span>User management system</div>
            <div class="feat-item"><span class="feat-dot"></span>CSV / Excel export tools</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-footer">🔒 Secure & Encrypted &nbsp;|&nbsp; LeadScore Pro v2.0 &nbsp;|&nbsp; © 2024</div>', unsafe_allow_html=True)

# ============================================================================
# INITIALIZE
# ============================================================================
init_database()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# Inject main theme
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR (Navy theme)
# ============================================================================
with st.sidebar:
    user_stats = get_user_stats(st.session_state.user['id'])
    uname = st.session_state.user['username']
    role = st.session_state.user['role']
    initials = uname[:2].upper()

    st.markdown(f"""
    <div class="sidebar-profile">
        <div class="sidebar-avatar">{initials}</div>
        <div class="sidebar-name">{uname}</div>
        <div class="sidebar-role">{role}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    st.markdown("""
    <div class="nav-item active"><span class="nav-icon">🏠</span> Home</div>
    <div class="nav-item"><span class="nav-icon">📊</span> Dashboard</div>
    <div class="nav-item"><span class="nav-icon">🔥</span> Hot Leads</div>
    <div class="nav-item"><span class="nav-icon">📈</span> Analytics</div>
    <div class="nav-item"><span class="nav-icon">💾</span> Export</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Data Source")
    upload_option = st.radio("Select:", ["Default Dataset", "Upload Custom File"], label_visibility="collapsed")
    if upload_option == "Upload Custom File":
        uploaded_file = st.file_uploader("Upload Excel", type=['xlsx', 'xls'])
        data_path = uploaded_file
    else:
        data_path = "5000_rental_crm_leads.xlsx"

    st.markdown("---")
    train_button = st.button("▶  Train & Score", use_container_width=True)
    st.markdown("---")

    st.markdown("### Your Stats")
    st.markdown(f"""
    <div style="padding: 0 4px;">
        <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #1e3660;">
            <span style="font-size:0.8rem;color:#9ab0cc;">Scorings</span>
            <span class="stat-pill">{user_stats['total_scorings']}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #1e3660;">
            <span style="font-size:0.8rem;color:#9ab0cc;">Leads Scored</span>
            <span class="stat-pill">{user_stats['total_leads']:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;">
            <span style="font-size:0.8rem;color:#9ab0cc;">Logins</span>
            <span class="stat-pill">{user_stats['total_logins']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("⏏  Logout", use_container_width=True, key="logout_btn"):
        logout_user(st.session_state.user['id'])
        log_usage(st.session_state.user['id'], 'logout')
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()

# ============================================================================
# ADMIN DASHBOARD
# ============================================================================
if st.session_state.user['role'] == 'admin':

    # Top bar
    sys_stats = get_system_stats()
    st.markdown(f"""
    <div class="page-topbar">
        <div>
            <p class="page-topbar-title">Admin Command Center</p>
            <p class="page-topbar-sub">{datetime.now().strftime("%A, %B %d %Y")} &nbsp;·&nbsp; Logged in as <b>{uname}</b></p>
        </div>
        <span class="topbar-badge">ADMIN</span>
    </div>
    """, unsafe_allow_html=True)

    # System stat cards
    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (col1, "👥", sys_stats['total_users'], "Total Users", "#3b82f6"),
        (col2, "🟢", sys_stats['currently_online'], "Online Now", "#10b981"),
        (col3, "📊", sys_stats['total_scorings'], "Scorings", "#f97316"),
        (col4, "📄", f"{sys_stats['total_leads']:,}", "Total Leads", "#8b5cf6"),
        (col5, "🕒", sys_stats['today_logins'], "Today Logins", "#ec4899"),
    ]
    for col, icon, val, label, accent in cards:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-accent" style="background:{accent};"></div>
                <div class="stat-card-icon" style="background:{accent}18;">{icon}</div>
                <div class="stat-card-value">{val}</div>
                <div class="stat-card-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    admin_tab1, admin_tab2 = st.tabs(["🎯 Lead Scoring", "👑 User Management"])

    # ── Admin Lead Scoring Tab ──
    with admin_tab1:
        if train_button and data_path:
            with st.spinner("Loading data..."):
                df = load_data(data_path)
            if df is not None:
                with st.expander("Dataset Preview", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rows", f"{len(df):,}")
                    c2.metric("Columns", len(df.columns))
                    c3.metric("Missing", df.isnull().sum().sum())
                    c4.metric("Memory", f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
                    st.dataframe(df.head(10), use_container_width=True)
                try:
                    model, scored_df, features, accuracy, roc_auc = train_model(df)
                    st.session_state.update({'model': model, 'scored_df': scored_df,
                                             'features': features, 'accuracy': accuracy, 'roc_auc': roc_auc})
                    log_usage(st.session_state.user['id'], 'score_leads', 'Admin scoring', len(scored_df))
                    st.success("Model trained & leads scored!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {e}")

        if 'scored_df' in st.session_state:
            df = st.session_state['scored_df']
            accuracy = st.session_state.get('accuracy', 0)
            roc_auc = st.session_state.get('roc_auc', None)

            t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "🔥 Priority Leads", "📈 Analytics", "📋 All Leads", "💾 Export"])

            with t1:
                hot = len(df[df['lead_category'] == 'Hot'])
                warm = len(df[df['lead_category'] == 'Warm'])
                cold = len(df[df['lead_category'] == 'Cold'])
                total = len(df)

                st.markdown('<div class="section-card"><p class="section-card-title">Lead Summary</p><p class="section-card-sub">Current scoring results</p>', unsafe_allow_html=True)
                c1, c2, c3, c4, c5 = st.columns(5)
                summary_cards = [
                    (c1, "📊", total, "Total Leads", "#0f2044"),
                    (c2, "🔥", hot, f"Hot ({hot/total*100:.0f}%)", "#dc2626"),
                    (c3, "🌡️", warm, f"Warm ({warm/total*100:.0f}%)", "#ea580c"),
                    (c4, "❄️", cold, f"Cold ({cold/total*100:.0f}%)", "#2563eb"),
                    (c5, "⭐", f"{df['lead_score'].mean():.1f}", "Avg Score", "#7c3aed"),
                ]
                for col, icon, val, label, accent in summary_cards:
                    with col:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-card-accent" style="background:{accent};"></div>
                            <div class="stat-card-icon" style="background:{accent}18;">{icon}</div>
                            <div class="stat-card-value">{val}</div>
                            <div class="stat-card-label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown('<div class="section-card"><p class="section-card-title">Model Performance</p>', unsafe_allow_html=True)
                    g1, g2, g3 = st.columns(3)
                    with g1:
                        fig = create_donut_chart(accuracy*100, "Accuracy", "#0f2044", "#f0f4f8")
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    with g2:
                        if roc_auc:
                            fig = create_donut_chart(roc_auc*100, "ROC AUC", "#f97316", "#fff7ed")
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    with g3:
                        conv = hot/total*100
                        fig = create_donut_chart(conv, "Hot %", "#dc2626", "#fee2e2")
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_r:
                    st.markdown('<div class="section-card"><p class="section-card-title">Category Breakdown</p>', unsafe_allow_html=True)
                    cat_counts = df['lead_category'].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=cat_counts.index, values=cat_counts.values, hole=0.55,
                        marker=dict(colors=['#dc2626', '#ea580c', '#2563eb']),
                        textinfo='label+percent',
                        textfont=dict(size=12, family='DM Sans'),
                        showlegend=False
                    )])
                    fig_pie.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=10),
                                          paper_bgcolor='rgba(0,0,0,0)', font={'family': 'DM Sans', 'color': '#0f2044'})
                    st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card"><p class="section-card-title">Score Distribution</p><p class="section-card-sub">Number of leads at each score range</p>', unsafe_allow_html=True)
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=df['lead_score'], nbinsx=20,
                                                 marker=dict(color='#0f2044', opacity=0.85)))
                fig_hist.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=10),
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis=dict(title='Score', color='#64748b', gridcolor='#f1f5f9'),
                                        yaxis=dict(title='Count', color='#64748b', gridcolor='#f1f5f9'),
                                        font={'family': 'DM Sans', 'color': '#0f2044'})
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

            with t2:
                st.markdown('<div class="section-card"><p class="section-card-title">Priority Leads</p><p class="section-card-sub">Filter and view your best leads</p>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    cat_filter = st.multiselect("Category", ['Hot', 'Warm', 'Cold'], default=['Hot'])
                with c2:
                    min_score = st.slider("Min Score", 0, 100, 70)
                with c3:
                    show_n = st.number_input("Show Top", 10, 100, 20, 10)
                filtered = df[(df['lead_category'].isin(cat_filter)) & (df['lead_score'] >= min_score)]
                display_cols = ['lead_id', 'name', 'lead_score', 'lead_category']
                opt = [c for c in ['source', 'budget_mid', 'preferred_area', 'total_interactions'] if c in filtered.columns]
                if opt:
                    sel = st.multiselect("Extra Columns", opt, opt[:2])
                    display_cols += sel
                top = filtered.nlargest(show_n, 'lead_score')[display_cols]
                st.dataframe(top, use_container_width=True, height=500)
                c1, c2, c3 = st.columns(3)
                c1.metric("Filtered", len(filtered))
                c2.metric("Avg Score", f"{filtered['lead_score'].mean():.1f}" if len(filtered) else "—")
                c3.metric("Max Score", filtered['lead_score'].max() if len(filtered) else "—")
                st.markdown('</div>', unsafe_allow_html=True)

            with t3:
                st.markdown('<div class="section-card"><p class="section-card-title">Analytics</p>', unsafe_allow_html=True)
                if 'source' in df.columns:
                    src = df.groupby('source').agg({'lead_score': ['mean', 'count'],
                                                     'lead_category': lambda x: (x=='Hot').sum()}).round(2)
                    src.columns = ['Avg Score', 'Count', 'Hot Leads']
                    src = src.sort_values('Avg Score', ascending=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.dataframe(src, use_container_width=True)
                    with c2:
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(x=src.index, y=src['Avg Score'],
                                                  marker=dict(color='#0f2044', opacity=0.85),
                                                  text=src['Avg Score'].round(1), textposition='outside'))
                        fig_bar.update_layout(title="Avg Score by Source", height=320,
                                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                               font={'family': 'DM Sans', 'color': '#0f2044'},
                                               xaxis=dict(color='#64748b'), yaxis=dict(color='#64748b', gridcolor='#f1f5f9'))
                        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                if 'budget_mid' in df.columns:
                    fig_sc = px.scatter(df.dropna(subset=['budget_mid']), x='budget_mid', y='lead_score',
                                        color='lead_category',
                                        color_discrete_map={'Hot': '#dc2626', 'Warm': '#ea580c', 'Cold': '#2563eb'},
                                        size='total_interactions' if 'total_interactions' in df.columns else None)
                    fig_sc.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font={'family': 'DM Sans', 'color': '#0f2044'},
                                          xaxis=dict(color='#64748b', gridcolor='#f1f5f9'),
                                          yaxis=dict(color='#64748b', gridcolor='#f1f5f9'))
                    st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

            with t4:
                st.markdown('<div class="section-card"><p class="section-card-title">All Leads</p>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1: search = st.text_input("Search", placeholder="Name or ID")
                with c2: score_rng = st.slider("Score Range", 0, 100, (0, 100))
                with c3: sort_col = st.selectbox("Sort", ['lead_score', 'lead_id', 'name'])
                with c4: sort_ord = st.radio("Order", ['Desc', 'Asc'])
                fdf = df.copy()
                if search:
                    fdf = fdf[fdf['name'].str.contains(search, case=False, na=False) |
                              fdf['lead_id'].astype(str).str.contains(search, case=False)]
                fdf = fdf[(fdf['lead_score'] >= score_rng[0]) & (fdf['lead_score'] <= score_rng[1])]
                fdf = fdf.sort_values(sort_col, ascending=(sort_ord == 'Asc'))
                st.info(f"{len(fdf):,} of {len(df):,} leads shown")
                st.dataframe(fdf, use_container_width=True, height=500)
                st.markdown('</div>', unsafe_allow_html=True)

            with t5:
                st.markdown('<div class="section-card"><p class="section-card-title">Export</p><p class="section-card-sub">Download your scored leads</p>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button("📄 Download CSV", df.to_csv(index=False).encode('utf-8'),
                                       'scored_leads.csv', 'text/csv', use_container_width=True)
                with c2:
                    @st.cache_data
                    def to_excel(d):
                        out = BytesIO()
                        with pd.ExcelWriter(out, engine='openpyxl') as w:
                            d.to_excel(w, index=False)
                        return out.getvalue()
                    st.download_button("📊 Download Excel", to_excel(df), 'scored_leads.xlsx',
                                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                       use_container_width=True)
                with c3:
                    hot_df = df[df['lead_category'] == 'Hot']
                    st.download_button("🔥 Hot Leads Only", hot_df.to_csv(index=False).encode('utf-8'),
                                       'hot_leads.csv', 'text/csv', use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="section-card" style="text-align:center; padding: 60px 20px;">
                <div style="font-size: 3rem; margin-bottom: 16px;">🎯</div>
                <h3 style="color: #0f2044; font-size: 1.3rem; margin-bottom: 8px;">Ready to Score Leads</h3>
                <p style="color: #94a3b8; font-size: 0.9rem;">Select a data source in the sidebar and click <b>Train & Score</b> to begin.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Admin User Management Tab ──
    with admin_tab2:
        st.markdown("""
        <div class="page-topbar" style="margin-bottom: 20px;">
            <div>
                <p class="page-topbar-title">User Management</p>
                <p class="page-topbar-sub">Create, manage, and monitor all users</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ut1, ut2, ut3, ut4 = st.tabs(["🟢 Live Users", "➕ Create User", "👥 All Users", "📊 Activity Log"])

        with ut1:
            st.markdown('<div class="section-card"><p class="section-card-title">Currently Online</p>', unsafe_allow_html=True)
            if st.button("Refresh", key="admin_refresh"):
                st.rerun()
            active = get_currently_logged_in_users()
            if active:
                st.success(f"{len(active)} user(s) online right now")
                for u in active:
                    uid, uname_, email_, login_t, urole = u
                    ustats = get_user_stats(uid)
                    st.markdown(f"""
                    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px 18px; margin:8px 0;">
                        <span class="dot-online"></span>
                        <b style="color:#0f2044;">{uname_}</b>
                        <span style="color:#64748b; font-size:0.8rem; margin-left:8px;">({urole})</span><br>
                        <small style="color:#64748b;">📧 {email_ or 'N/A'} &nbsp;|&nbsp; 🕒 {login_t}</small><br>
                        <small style="color:#64748b;">📊 {ustats['total_scorings']} scorings &nbsp;|&nbsp; 📄 {ustats['total_leads']:,} leads</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No users currently online")
            st.markdown('</div>', unsafe_allow_html=True)

        with ut2:
            st.markdown('<div class="section-card"><p class="section-card-title">Create New User</p>', unsafe_allow_html=True)
            with st.form("create_user_form"):
                ca, cb = st.columns(2)
                with ca:
                    nu = st.text_input("Username *")
                    ne = st.text_input("Email")
                with cb:
                    np_ = st.text_input("Password *", type="password")
                    cp = st.text_input("Confirm Password *", type="password")
                if st.form_submit_button("Create User", type="primary", use_container_width=True):
                    if nu and np_ == cp and len(np_) >= 6:
                        if create_user_by_admin(nu, np_, ne):
                            st.success(f"User `{nu}` created successfully!")
                        else:
                            st.error("Username already exists.")
                    elif len(np_) < 6:
                        st.error("Password must be ≥ 6 characters")
                    else:
                        st.error("Passwords don't match or fields are empty")
            st.markdown('</div>', unsafe_allow_html=True)

        with ut3:
            st.markdown('<div class="section-card"><p class="section-card-title">All Registered Users</p>', unsafe_allow_html=True)
            users = get_all_users()
            if users:
                user_data = [{'ID': u[0], 'Username': u[1], 'Email': u[2] or 'N/A',
                              'Created': u[3], 'Last Login': u[4] or 'Never',
                              'Status': '🟢 Active' if u[5] else '🔴 Inactive', 'Role': u[6]}
                             for u in users]
                st.dataframe(pd.DataFrame(user_data), use_container_width=True, height=400)
            st.markdown("---")
            st.markdown("**User Actions**")
            ca, cb, cc = st.columns(3)
            with ca: uid_a = st.number_input("User ID", min_value=1, step=1)
            with cb: action_t = st.selectbox("Action", ["Enable", "Disable", "Delete"])
            with cc:
                st.write("")
                if st.button("Execute", type="primary"):
                    if uid_a != 1:
                        if action_t == "Enable":
                            toggle_user_status(uid_a, 1); st.success("User enabled!"); time.sleep(1); st.rerun()
                        elif action_t == "Disable":
                            toggle_user_status(uid_a, 0); st.warning("User disabled!"); time.sleep(1); st.rerun()
                        elif action_t == "Delete":
                            delete_user(uid_a); st.error("User deleted!"); time.sleep(1); st.rerun()
                    else:
                        st.error("Cannot modify admin account")
            st.markdown('</div>', unsafe_allow_html=True)

        with ut4:
            st.markdown('<div class="section-card"><p class="section-card-title">Activity Log</p><p class="section-card-sub">Last 100 actions across all users</p>', unsafe_allow_html=True)
            all_acts = get_all_user_activities()
            if all_acts:
                act_data = [{'Username': a[0], 'Action': a[1], 'Details': a[2] or '—',
                             'Leads': a[3] or '—', 'Timestamp': a[4]} for a in all_acts]
                st.dataframe(pd.DataFrame(act_data), use_container_width=True, height=500)
            else:
                st.info("No activities logged yet")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# USER DASHBOARD
# ============================================================================
else:
    st.markdown(f"""
    <div class="page-topbar">
        <div>
            <p class="page-topbar-title">Lead Scoring Dashboard</p>
            <p class="page-topbar-sub">{datetime.now().strftime("%A, %B %d %Y")} &nbsp;·&nbsp; Welcome back, <b>{uname}</b></p>
        </div>
        <span class="topbar-badge">USER</span>
    </div>
    """, unsafe_allow_html=True)

    if train_button and data_path:
        with st.spinner("Loading data..."):
            df = load_data(data_path)
        if df is not None:
            with st.expander("Dataset Preview", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{len(df):,}")
                c2.metric("Columns", len(df.columns))
                c3.metric("Missing", df.isnull().sum().sum())
                c4.metric("Memory", f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
                st.dataframe(df.head(10), use_container_width=True)
            try:
                model, scored_df, features, accuracy, roc_auc = train_model(df)
                st.session_state.update({'model': model, 'scored_df': scored_df,
                                         'features': features, 'accuracy': accuracy, 'roc_auc': roc_auc})
                log_usage(st.session_state.user['id'], 'score_leads', 'User scoring', len(scored_df))
                st.success("Scoring complete!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

    if 'scored_df' in st.session_state:
        df = st.session_state['scored_df']
        accuracy = st.session_state.get('accuracy', 0)
        roc_auc = st.session_state.get('roc_auc', None)

        t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "🔥 Priority", "📈 Analytics", "📋 All Leads", "💾 Export"])

        with t1:
            hot = len(df[df['lead_category'] == 'Hot'])
            warm = len(df[df['lead_category'] == 'Warm'])
            cold = len(df[df['lead_category'] == 'Cold'])
            total = len(df)

            c1, c2, c3, c4, c5 = st.columns(5)
            cards = [
                (c1, "📊", total, "Total Leads", "#0f2044"),
                (c2, "🔥", hot, f"Hot ({hot/total*100:.0f}%)", "#dc2626"),
                (c3, "🌡️", warm, f"Warm ({warm/total*100:.0f}%)", "#ea580c"),
                (c4, "❄️", cold, f"Cold ({cold/total*100:.0f}%)", "#2563eb"),
                (c5, "⭐", f"{df['lead_score'].mean():.1f}", "Avg Score", "#7c3aed"),
            ]
            for col, icon, val, label, accent in cards:
                with col:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-card-accent" style="background:{accent};"></div>
                        <div class="stat-card-icon" style="background:{accent}18;">{icon}</div>
                        <div class="stat-card-value">{val}</div>
                        <div class="stat-card-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown('<div class="section-card"><p class="section-card-title">Model Performance</p>', unsafe_allow_html=True)
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.plotly_chart(create_donut_chart(accuracy*100, "Accuracy", "#0f2044", "#f0f4f8"),
                                    use_container_width=True, config={'displayModeBar': False})
                with g2:
                    if roc_auc:
                        st.plotly_chart(create_donut_chart(roc_auc*100, "ROC AUC", "#f97316", "#fff7ed"),
                                        use_container_width=True, config={'displayModeBar': False})
                with g3:
                    st.plotly_chart(create_donut_chart(hot/total*100, "Hot %", "#dc2626", "#fee2e2"),
                                    use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown('<div class="section-card"><p class="section-card-title">Category Breakdown</p>', unsafe_allow_html=True)
                cat_counts = df['lead_category'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=cat_counts.index, values=cat_counts.values, hole=0.55,
                    marker=dict(colors=['#dc2626', '#ea580c', '#2563eb']),
                    textinfo='label+percent', textfont=dict(size=12, family='DM Sans'), showlegend=False
                )])
                fig_pie.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=10),
                                      paper_bgcolor='rgba(0,0,0,0)', font={'family': 'DM Sans', 'color': '#0f2044'})
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><p class="section-card-title">Score Distribution</p>', unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df['lead_score'], nbinsx=20,
                                             marker=dict(color='#0f2044', opacity=0.85)))
            fig_hist.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=10),
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(color='#64748b', gridcolor='#f1f5f9'),
                                    yaxis=dict(color='#64748b', gridcolor='#f1f5f9'),
                                    font={'family': 'DM Sans'})
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with t2:
            st.markdown('<div class="section-card"><p class="section-card-title">Priority Leads</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1: cat_filter = st.multiselect("Category", ['Hot', 'Warm', 'Cold'], default=['Hot'])
            with c2: min_score = st.slider("Min Score", 0, 100, 70)
            with c3: show_n = st.number_input("Show", 10, 100, 20, 10)
            filtered = df[(df['lead_category'].isin(cat_filter)) & (df['lead_score'] >= min_score)]
            st.dataframe(filtered.nlargest(show_n, 'lead_score'), use_container_width=True, height=500)
            st.markdown('</div>', unsafe_allow_html=True)

        with t3:
            st.markdown('<div class="section-card"><p class="section-card-title">Analytics</p>', unsafe_allow_html=True)
            if 'source' in df.columns:
                src = df.groupby('source')['lead_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=src.index, y=src['mean'],
                                          marker=dict(color='#0f2044', opacity=0.85),
                                          text=src['mean'].round(1), textposition='outside'))
                fig_bar.update_layout(title="Avg Score by Source", height=320,
                                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       font={'family': 'DM Sans', 'color': '#0f2044'},
                                       xaxis=dict(color='#64748b'), yaxis=dict(color='#64748b', gridcolor='#f1f5f9'))
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with t4:
            st.markdown('<div class="section-card"><p class="section-card-title">All Leads</p>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=500)
            st.markdown('</div>', unsafe_allow_html=True)

        with t5:
            st.markdown('<div class="section-card"><p class="section-card-title">Export</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("📄 CSV", df.to_csv(index=False).encode('utf-8'),
                                   'leads.csv', 'text/csv', use_container_width=True)
            with c2:
                @st.cache_data
                def to_excel2(d):
                    out = BytesIO()
                    with pd.ExcelWriter(out, engine='openpyxl') as w:
                        d.to_excel(w, index=False)
                    return out.getvalue()
                st.download_button("📊 Excel", to_excel2(df), 'leads.xlsx',
                                   'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   use_container_width=True)
            with c3:
                st.download_button("🔥 Hot Only", df[df['lead_category']=='Hot'].to_csv(index=False).encode('utf-8'),
                                   'hot_leads.csv', 'text/csv', use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="section-card" style="text-align:center; padding: 60px 20px;">
            <div style="font-size: 3rem; margin-bottom: 16px;">🎯</div>
            <h3 style="color: #0f2044; font-size: 1.3rem; margin-bottom: 8px;">Ready to Score Your Leads</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">Select a data source from the sidebar and click <b>Train & Score</b> to get started.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center; padding: 16px 0; margin-top: 8px; border-top: 1px solid #e8edf2;">
    <p style="font-size:0.75rem; color:#94a3b8; margin:0;">
        LeadScore Pro v2.0 &nbsp;·&nbsp; Logged in as <b style="color:#0f2044;">{st.session_state.user['username']}</b> &nbsp;·&nbsp; {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </p>
</div>
""", unsafe_allow_html=True)
