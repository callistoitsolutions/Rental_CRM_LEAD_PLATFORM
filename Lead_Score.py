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
    page_title="AI Lead Scoring Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database():
    """Initialize database with migration support"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  email TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  last_login TIMESTAMP,
                  is_active BOOLEAN DEFAULT 1,
                  role TEXT DEFAULT 'user')''')
    
    # Create usage_logs table
    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  action TEXT,
                  details TEXT,
                  leads_scored INTEGER,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  logout_time TIMESTAMP,
                  is_active BOOLEAN DEFAULT 1,
                  session_token TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Migration: Add missing columns
    try:
        c.execute("PRAGMA table_info(sessions)")
        columns = [column[1] for column in c.fetchall()]
        
        if 'is_active' not in columns:
            c.execute("ALTER TABLE sessions ADD COLUMN is_active BOOLEAN DEFAULT 1")
            conn.commit()
            
        if 'session_token' not in columns:
            c.execute("ALTER TABLE sessions ADD COLUMN session_token TEXT")
            conn.commit()
    except Exception as e:
        pass
    
    # Create admin user if not exists
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
    """Verify and login user"""
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
            
            return {
                'id': user[0], 
                'username': user[1], 
                'role': user[2], 
                'is_active': user[3], 
                'session_token': session_token
            }
        
        conn.close()
        return None
    except Exception as e:
        conn.close()
        return None

def create_user_by_admin(username, password, email):
    """Admin creates user"""
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
    """Logout user"""
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
    """Log activity"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO usage_logs (user_id, action, details, leads_scored) VALUES (?, ?, ?, ?)",
              (user_id, action, details, leads_scored))
    conn.commit()
    conn.close()

def get_user_stats(user_id):
    """Get user stats"""
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
    """Get all users"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, username, email, created_at, last_login, is_active, role FROM users ORDER BY created_at DESC")
    users = c.fetchall()
    conn.close()
    return users

def get_currently_logged_in_users():
    """Get currently logged in users"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT u.id, u.username, u.email, s.login_time, u.role
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.is_active = 1
            ORDER BY s.login_time DESC
        """)
        active_users = c.fetchall()
    except sqlite3.OperationalError:
        active_users = []
    conn.close()
    return active_users

def get_user_activity_details(user_id):
    """Get detailed activity for a specific user"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        SELECT action, details, leads_scored, timestamp
        FROM usage_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 20
    """, (user_id,))
    activities = c.fetchall()
    conn.close()
    return activities

def get_all_user_activities():
    """Get all activities from all users"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        SELECT u.username, l.action, l.details, l.leads_scored, l.timestamp
        FROM usage_logs l
        JOIN users u ON l.user_id = u.id
        ORDER BY l.timestamp DESC
        LIMIT 100
    """)
    activities = c.fetchall()
    conn.close()
    return activities

def get_system_stats():
    """Get system stats"""
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
    return {
        'total_users': total_users,
        'currently_online': currently_online,
        'total_scorings': total_scorings,
        'total_leads': total_leads,
        'today_logins': today_logins
    }

def toggle_user_status(user_id, is_active):
    """Enable/disable user"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("UPDATE users SET is_active = ? WHERE id = ?", (is_active, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id):
    """Delete user"""
    conn = sqlite3.connect('lead_scoring.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def map_probability_to_category(prob_score):
    """Map probability (0-100) to category label."""
    if prob_score >= 70:
        return "Hot"
    elif prob_score >= 40:
        return "Warm"
    else:
        return "Cold"

@st.cache_data
def load_data(file_path):
    """Load data from Excel file"""
    try:
        if isinstance(file_path, str):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_resource
def train_model(df):
    """Train RandomForest model with progress tracking"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Feature engineering
    status_text.markdown("🔧 **Step 1/5:** Feature Engineering...")
    progress_bar.progress(20)
    
    if "budget_min" in df.columns and "budget_max" in df.columns:
        df["budget_mid"] = df[["budget_min", "budget_max"]].mean(axis=1)
    elif "budget" in df.columns:
        df["budget_mid"] = pd.to_numeric(df["budget"], errors='coerce')
    else:
        df["budget_mid"] = np.nan

    # Budget match feature
    if df["budget_mid"].notna().any():
        min_b, max_b = df["budget_mid"].min(), df["budget_mid"].max()
        if min_b == max_b or pd.isna(min_b) or pd.isna(max_b):
            df["budget_match"] = 1.0
        else:
            df["budget_match"] = (df["budget_mid"] - min_b) / (max_b - min_b)
    else:
        df["budget_match"] = 0.5

    # Area match feature
    if "preferred_area" in df.columns:
        area_freq = df["preferred_area"].fillna("unknown").value_counts(normalize=True)
        df["area_match"] = df["preferred_area"].fillna("unknown").map(area_freq).fillna(0.5)
    else:
        df["area_match"] = 0.5

    # Behavior scores
    beh_cols = ["views_count", "avg_view_time_sec", "saved_properties", "repeated_visits"]
    for c in beh_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Normalize behavior columns
    for c in beh_cols:
        mx = df[c].max()
        if mx > 0:
            df[c + "_norm"] = df[c] / mx
        else:
            df[c + "_norm"] = 0.0

    # Engagement score
    df["engagement_score"] = (
        0.4 * df["views_count_norm"] +
        0.2 * df["avg_view_time_sec_norm"] +
        0.25 * df["saved_properties_norm"] +
        0.15 * df["repeated_visits_norm"]
    )

    # Interaction features
    inter_cols = ["whatsapp_clicks", "call_clicks", "chat_messages"]
    for c in inter_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["total_interactions"] = df[inter_cols].sum(axis=1)

    # Recency features
    if "last_active_time" in df.columns:
        df["last_active_time"] = pd.to_datetime(df["last_active_time"], errors="coerce")
        now = pd.Timestamp.now()
        df["days_since_active"] = (now - df["last_active_time"]).dt.days.fillna(999)
        df["recency_score"] = 1 / (1 + df["days_since_active"])
    else:
        df["recency_score"] = 0.0

    # Prepare features
    status_text.markdown("📊 **Step 2/5:** Preparing Features...")
    progress_bar.progress(40)
    
    feature_cols = [
        "budget_match", "area_match", "engagement_score",
        "total_interactions", "recency_score",
    ]
    
    if "source" in df.columns:
        feature_cols.append("source")
    if "bhk" in df.columns:
        feature_cols.append("bhk")

    X = df[feature_cols].copy()

    # Prepare target
    y = None
    if "converted" in df.columns:
        y = pd.to_numeric(df["converted"], errors="coerce")

    # Handle missing labels
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

    # Build preprocessing pipeline
    status_text.markdown("🔨 **Step 3/5:** Building ML Pipeline...")
    progress_bar.progress(60)
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if num_cols:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_transformer, num_cols))

    if cat_cols:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # RandomForest model
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("rf", rf)
    ])

    # Train/test split
    status_text.markdown("🎯 **Step 4/5:** Training Model...")
    progress_bar.progress(80)
    
    stratify_y = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_y
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = None
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            pass

    # Score all leads
    status_text.markdown("✨ **Step 5/5:** Scoring All Leads...")
    progress_bar.progress(100)
    
    df_scored = df.copy()
    lead_probability = pipeline.predict_proba(X)[:, 1]
    df_scored.loc[X.index, "lead_score"] = (lead_probability * 100).round(0).astype(int)
    df_scored["lead_score"] = df_scored["lead_score"].fillna(0).astype(int)
    df_scored["lead_category"] = df_scored["lead_score"].apply(map_probability_to_category)
    
    status_text.markdown("✅ **Model Training Complete!**")
    progress_bar.progress(100)
    
    return pipeline, df_scored, feature_cols, accuracy, roc_auc

def create_gauge_chart(value, title, color):
    """Create a professional gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20, 'color': '#1e293b', 'family': 'Inter'}},
        number={'font': {'size': 40, 'color': color, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 40], 'color': '#dbeafe'},
                {'range': [40, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    return fig

# ============================================================================
# LOGIN PAGE
# ============================================================================

def show_login_page():
    """Professional login page with modern design"""
    
    # Creative CSS for login page
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        .login-container {
            max-width: 480px;
            margin: 80px auto;
            padding: 0;
            background: white;
            border-radius: 24px;
            box-shadow: 0 30px 80px rgba(0,0,0,0.3);
            overflow: hidden;
            animation: slideUp 0.6s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .login-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 30px;
            text-align: center;
            color: white;
            position: relative;
            overflow: hidden;
        }
        
        .login-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.3; }
        }
        
        .login-icon {
            font-size: 80px;
            margin-bottom: 15px;
            display: inline-block;
            animation: float 3s ease-in-out infinite;
            position: relative;
            z-index: 1;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
        }
        
        .login-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            position: relative;
            z-index: 1;
            letter-spacing: -0.5px;
        }
        
        .login-subtitle {
            font-size: 1rem;
            opacity: 0.95;
            margin-top: 8px;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }
        
        .login-body {
            padding: 40px 35px;
        }
        
        .input-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 8px;
            display: block;
        }
        
        .stTextInput > div > div > input {
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: #f8fafc;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        .login-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 16px 24px;
            font-size: 1.1rem;
            font-weight: 700;
            width: 100%;
            margin-top: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        .login-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
        }
        
        .demo-button {
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
            border-radius: 12px;
            padding: 16px 24px;
            font-size: 1.1rem;
            font-weight: 700;
            width: 100%;
            margin-top: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .demo-button:hover {
            background: #f8fafc;
            transform: translateY(-2px);
        }
        
        .feature-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
            padding: 20px;
            border-radius: 16px;
            margin-top: 30px;
            border: 2px solid #e0e7ff;
        }
        
        .feature-title {
            font-size: 1rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .feature-item {
            font-size: 0.9rem;
            color: #475569;
            margin: 8px 0;
            padding-left: 24px;
            position: relative;
        }
        
        .feature-item::before {
            content: '✓';
            position: absolute;
            left: 0;
            color: #10b981;
            font-weight: 800;
            font-size: 1.1rem;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .login-container {
                margin: 20px;
                max-width: 100%;
            }
            
            .login-title {
                font-size: 1.8rem;
            }
            
            .login-icon {
                font-size: 60px;
            }
            
            .login-body {
                padding: 30px 25px;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2.5, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Header Section
        st.markdown('''
        <div class="login-header">
            <div class="login-icon">🎯</div>
            <h1 class="login-title">Lead Scoring Pro</h1>
            <p class="login-subtitle">AI-Powered Lead Intelligence Platform</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Body Section
        st.markdown('<div class="login-body">', unsafe_allow_html=True)
        
        st.markdown('<label class="input-label">👤 Username</label>', unsafe_allow_html=True)
        username = st.text_input("", key="login_username", placeholder="Enter your username", label_visibility="collapsed")
        
        st.markdown('<label class="input-label" style="margin-top: 20px;">🔒 Password</label>', unsafe_allow_html=True)
        password = st.text_input("", type="password", key="login_password", placeholder="Enter your password", label_visibility="collapsed")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Login", use_container_width=True, type="primary", key="login_btn"):
                if username and password:
                    user = verify_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        log_usage(user['id'], 'login')
                        st.success(f"✅ Welcome back, {username}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        with col_btn2:
            if st.button("🔑 Demo Access", use_container_width=True, key="demo_btn"):
                with st.expander("📌 Demo Credentials", expanded=True):
                    st.markdown("""
                        **Admin Account:**
                        - Username: `admin`
                        - Password: `admin123`
                        
                        **Features:**
                        - Full system access
                        - User management
                        - Analytics dashboard
                    """)
        
        # Features Section
        st.markdown('''
        <div class="feature-card">
            <div class="feature-title">
                <span>⭐</span>
                <span>Platform Features</span>
            </div>
            <div class="feature-item">AI-powered lead scoring engine</div>
            <div class="feature-item">Real-time analytics dashboard</div>
            <div class="feature-item">Advanced user management</div>
            <div class="feature-item">Export & reporting tools</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close login-body
        st.markdown('</div>', unsafe_allow_html=True)  # Close login-container
        
        # Footer
        st.markdown('''
        <div style="text-align: center; margin-top: 30px; color: white; font-size: 0.9rem; opacity: 0.9;">
            <p>🔒 Secure & Encrypted | ⚡ Fast & Reliable | 📱 Mobile Responsive</p>
            <p style="opacity: 0.7; font-size: 0.85rem;">© 2024 Lead Scoring Pro. All rights reserved.</p>
        </div>
        ''', unsafe_allow_html=True)

# ============================================================================
# INITIALIZE
# ============================================================================

init_database()

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Check login
if not st.session_state.logged_in:
    show_login_page()
    st.stop()

# ============================================================================
# MAIN APPLICATION CSS (After Login)
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        margin-bottom: 0;
        letter-spacing: -0.03em;
        text-shadow: 0 0 80px rgba(96, 165, 250, 0.5);
    }
    
    .sub-header {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
    }
    
    /* User Info Card */
    .user-info {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 2px solid rgba(59, 130, 246, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .user-info h3 {
        color: #60a5fa !important;
        margin: 0 0 12px 0;
        font-size: 1.3rem;
        font-weight: 800;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        height: 100%;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .metric-card-hot {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
        border-color: #ef4444;
    }
    
    .metric-card-warm {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.1) 100%);
        border-color: #f59e0b;
    }
    
    .metric-card-cold {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%);
        border-color: #3b82f6;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 700;
        color: #94a3b8;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .dataframe {
        border: none !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 16px !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 1px;
        border: none !important;
    }
    
    .dataframe tbody tr {
        background: rgba(255, 255, 255, 0.02) !important;
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        transform: scale(1.005);
    }
    
    .dataframe tbody td {
        color: #e2e8f0 !important;
        padding: 12px !important;
        border-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 10px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-radius: 12px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        font-weight: 700;
        color: #e2e8f0;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);
    }
    
    /* Section Headers */
    h3 {
        color: #e2e8f0 !important;
        font-weight: 800 !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
        display: inline-block;
        letter-spacing: -0.5px;
    }
    
    /* Metric Values */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 900;
        color: #e2e8f0;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        padding: 0.75rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* Activity Card */
    .activity-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.04) 100%);
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .activity-card:hover {
        transform: translateX(5px);
        border-left-color: #8b5cf6;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Online Indicator */
    .online-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-green 2s infinite;
        box-shadow: 0 0 10px #10b981;
    }
    
    @keyframes pulse-green {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.1); }
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR (Common for both Admin & User)
# ============================================================================

with st.sidebar:
    user_stats = get_user_stats(st.session_state.user['id'])
    
    st.markdown(f"""
    <div class='user-info'>
        <h3>👤 {st.session_state.user['username']}</h3>
        <p style='margin: 8px 0; opacity: 0.9;'>Role: <b style='color: #60a5fa;'>{st.session_state.user['role'].upper()}</b></p>
        <hr style='margin: 12px 0; opacity: 0.3;'>
        <p style='margin: 6px 0;'>📊 Scorings: <b>{user_stats['total_scorings']}</b></p>
        <p style='margin: 6px 0;'>📄 Leads: <b>{user_stats['total_leads']:,}</b></p>
        <p style='margin: 6px 0;'>🔑 Logins: <b>{user_stats['total_logins']}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 LOGOUT", use_container_width=True):
        logout_user(st.session_state.user['id'])
        log_usage(st.session_state.user['id'], 'logout')
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()
    
    st.markdown("---")
    
    # Stats Summary
    st.markdown("### 📊 Your Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%); 
                    padding: 16px; border-radius: 12px; text-align: center; border: 2px solid rgba(59, 130, 246, 0.3);'>
            <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;'>Scorings</div>
            <div style='font-size: 2rem; font-weight: 900; color: #60a5fa; margin: 4px 0;'>{user_stats['total_scorings']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%); 
                    padding: 16px; border-radius: 12px; text-align: center; border: 2px solid rgba(139, 92, 246, 0.3);'>
            <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;'>Total Leads</div>
            <div style='font-size: 2rem; font-weight: 900; color: #a78bfa; margin: 4px 0;'>{user_stats['total_leads']:,}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# ADMIN DASHBOARD
# ============================================================================

if st.session_state.user['role'] == 'admin':
    
    st.markdown('<div class="main-header">👑 ADMIN COMMAND CENTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Complete System Management & Lead Scoring Analytics</div>', unsafe_allow_html=True)
    
    # System Stats
    sys_stats = get_system_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👥 Total Users", sys_stats['total_users'], help="Active user accounts")
    with col2:
        st.metric("🟢 Online Now", sys_stats['currently_online'], help="Currently logged in")
    with col3:
        st.metric("📊 Total Scorings", sys_stats['total_scorings'], help="All-time scorings")
    with col4:
        st.metric("📄 Total Leads", f"{sys_stats['total_leads']:,}", help="All leads scored")
    with col5:
        st.metric("🕒 Today Logins", sys_stats['today_logins'], help="Logins today")
    
    st.markdown("---")
    
    # Main Admin Tabs
    admin_main_tab1, admin_main_tab2 = st.tabs([
        "🎯 LEAD SCORING DASHBOARD",
        "👑 USER MANAGEMENT"
    ])
    
    # ========================================================================
    # ADMIN TAB 1: LEAD SCORING DASHBOARD
    # ========================================================================
    
    with admin_main_tab1:
        st.markdown("## 🚀 AI Lead Scoring Engine")
        
        # Sidebar for data source
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📂 Data Source")
            
            upload_option = st.radio(
                "Select Source:",
                ["Use Default Dataset", "Upload Custom File"],
                help="Choose your data source"
            )
            
            if upload_option == "Upload Custom File":
                uploaded_file = st.file_uploader(
                    "Upload Excel File",
                    type=['xlsx', 'xls'],
                    help="Upload CRM leads file"
                )
                data_path = uploaded_file
            else:
                data_path = "5000_rental_crm_leads.xlsx"
            
            st.markdown("---")
            
            train_button = st.button(
                "🚀 TRAIN & SCORE",
                type="primary",
                use_container_width=True
            )
        
        # Main content
        if train_button and data_path:
            with st.spinner("🔄 Loading data..."):
                df = load_data(data_path)
            
            if df is not None:
                with st.expander("📊 Dataset Preview", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📝 Rows", f"{len(df):,}")
                    with col2:
                        st.metric("📋 Columns", len(df.columns))
                    with col3:
                        st.metric("❓ Missing", df.isnull().sum().sum())
                    with col4:
                        memory = df.memory_usage(deep=True).sum() / 1024**2
                        st.metric("💾 Memory", f"{memory:.2f} MB")
                    
                    st.dataframe(df.head(10), use_container_width=True)
                
                try:
                    model, scored_df, features, accuracy, roc_auc = train_model(df)
                    
                    st.session_state['model'] = model
                    st.session_state['scored_df'] = scored_df
                    st.session_state['features'] = features
                    st.session_state['accuracy'] = accuracy
                    st.session_state['roc_auc'] = roc_auc
                    
                    log_usage(st.session_state.user['id'], 'score_leads', 'Admin scoring', len(scored_df))
                    
                    st.success("✅ Model trained successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # Display results
        if 'scored_df' in st.session_state:
            df = st.session_state['scored_df']
            accuracy = st.session_state.get('accuracy', 0)
            roc_auc = st.session_state.get('roc_auc', None)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 DASHBOARD",
                "🔥 PRIORITY LEADS",
                "📈 ANALYTICS",
                "📋 ALL LEADS",
                "💾 EXPORT"
            ])
            
            with tab1:
                st.markdown("### 📊 Performance Dashboard")
                st.markdown("")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("📊 Total", f"{len(df):,}")
                with col2:
                    hot = len(df[df['lead_category'] == 'Hot'])
                    st.markdown(f"""
                        <div class="metric-card metric-card-hot">
                            <div style="font-size: 0.9rem; color: #fca5a5; font-weight: 700; text-transform: uppercase;">🔥 HOT</div>
                            <div style="font-size: 2.5rem; font-weight: 900; color: #ef4444; margin: 0.5rem 0;">{hot}</div>
                            <div style="font-size: 0.85rem; color: #fca5a5;">{hot/len(df)*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    warm = len(df[df['lead_category'] == 'Warm'])
                    st.markdown(f"""
                        <div class="metric-card metric-card-warm">
                            <div style="font-size: 0.9rem; color: #fcd34d; font-weight: 700; text-transform: uppercase;">🌡️ WARM</div>
                            <div style="font-size: 2.5rem; font-weight: 900; color: #f59e0b; margin: 0.5rem 0;">{warm}</div>
                            <div style="font-size: 0.85rem; color: #fcd34d;">{warm/len(df)*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col4:
                    cold = len(df[df['lead_category'] == 'Cold'])
                    st.markdown(f"""
                        <div class="metric-card metric-card-cold">
                            <div style="font-size: 0.9rem; color: #93c5fd; font-weight: 700; text-transform: uppercase;">❄️ COLD</div>
                            <div style="font-size: 2.5rem; font-weight: 900; color: #3b82f6; margin: 0.5rem 0;">{cold}</div>
                            <div style="font-size: 0.85rem; color: #93c5fd;">{cold/len(df)*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col5:
                    st.metric("⭐ Avg Score", f"{df['lead_score'].mean():.1f}")
                
                st.markdown("")
                st.markdown("---")
                
                st.markdown("### 🎯 Model Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    gauge = create_gauge_chart(accuracy*100, "Accuracy", "#3b82f6")
                    st.plotly_chart(gauge, use_container_width=True)
                
                with col2:
                    if roc_auc:
                        gauge = create_gauge_chart(roc_auc*100, "ROC AUC", "#8b5cf6")
                        st.plotly_chart(gauge, use_container_width=True)
                    else:
                        st.info("ROC AUC not available")
                
                with col3:
                    conversion = (df['lead_score'] > 70).sum() / len(df) * 100
                    gauge = create_gauge_chart(conversion, "Hot %", "#10b981")
                    st.plotly_chart(gauge, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 📊 Distribution Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    category_counts = df['lead_category'].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=category_counts.index,
                        values=category_counts.values,
                        hole=0.4,
                        marker=dict(colors=['#ef4444', '#f59e0b', '#3b82f6']),
                        textinfo='label+percent',
                        textfont=dict(size=14, family='Inter', color='white')
                    )])
                    fig_pie.update_layout(
                        title="Lead Categories",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter', 'color': '#e2e8f0'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=df['lead_score'],
                        nbinsx=20,
                        marker=dict(
                            color=df['lead_score'],
                            colorscale='Viridis'
                        )
                    ))
                    fig_hist.update_layout(
                        title="Score Distribution",
                        xaxis_title="Lead Score",
                        yaxis_title="Count",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter', 'color': '#e2e8f0'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with tab2:
                st.markdown("### 🔥 Top Priority Leads")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    category_filter = st.multiselect(
                        "Filter by Category",
                        ['Hot', 'Warm', 'Cold'],
                        default=['Hot']
                    )
                with col2:
                    min_score = st.slider("Minimum Score", 0, 100, 70)
                with col3:
                    show_count = st.number_input("Show Top", 10, 100, 20, 10)
                
                filtered = df[
                    (df['lead_category'].isin(category_filter)) & 
                    (df['lead_score'] >= min_score)
                ]
                
                display_cols = ['lead_id', 'name', 'lead_score', 'lead_category']
                optional_cols = ['source', 'budget_mid', 'preferred_area', 'total_interactions']
                available = [c for c in optional_cols if c in filtered.columns]
                
                if available:
                    selected = st.multiselect("Additional Columns", available, available[:2] if len(available) >= 2 else available)
                    display_cols.extend(selected)
                
                top_leads = filtered.nlargest(show_count, 'lead_score')[display_cols]
                
                def highlight_category(row):
                    if row['lead_category'] == 'Hot':
                        return ['background-color: rgba(239, 68, 68, 0.2)'] * len(row)
                    elif row['lead_category'] == 'Warm':
                        return ['background-color: rgba(245, 158, 11, 0.2)'] * len(row)
                    else:
                        return ['background-color: rgba(59, 130, 246, 0.2)'] * len(row)
                
                st.dataframe(top_leads.style.apply(highlight_category, axis=1), use_container_width=True, height=600)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Filtered", len(filtered))
                with col2:
                    st.metric("📊 Avg", f"{filtered['lead_score'].mean():.1f}")
                with col3:
                    st.metric("📈 Max", filtered['lead_score'].max())
            
            with tab3:
                st.markdown("### 📈 Advanced Analytics")
                
                if 'source' in df.columns:
                    st.markdown("#### Performance by Source")
                    source_stats = df.groupby('source').agg({
                        'lead_score': ['mean', 'count'],
                        'lead_category': lambda x: (x == 'Hot').sum()
                    }).round(2)
                    source_stats.columns = ['Avg Score', 'Count', 'Hot Leads']
                    source_stats = source_stats.sort_values('Avg Score', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(source_stats, use_container_width=True)
                    with col2:
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=source_stats.index,
                            y=source_stats['Avg Score'],
                            marker=dict(
                                color=source_stats['Avg Score'],
                                colorscale='Viridis'
                            ),
                            text=source_stats['Avg Score'].round(1),
                            textposition='outside'
                        ))
                        fig_bar.update_layout(
                            title="Avg Score by Source",
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'family': 'Inter', 'color': '#e2e8f0'}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                if 'budget_mid' in df.columns:
                    st.markdown("#### Score vs Budget")
                    fig_scatter = px.scatter(
                        df.dropna(subset=['budget_mid']),
                        x='budget_mid',
                        y='lead_score',
                        color='lead_category',
                        size='total_interactions' if 'total_interactions' in df.columns else None,
                        color_discrete_map={'Hot': '#ef4444', 'Warm': '#f59e0b', 'Cold': '#3b82f6'}
                    )
                    fig_scatter.update_layout(
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter', 'color': '#e2e8f0'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab4:
                st.markdown("### 📋 Complete Database")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    search = st.text_input("🔍 Search", placeholder="Name or ID")
                with col2:
                    score_range = st.slider("Score Range", 0, 100, (0, 100))
                with col3:
                    sort_by = st.selectbox("Sort by", ['lead_score', 'lead_id', 'name'])
                with col4:
                    sort_order = st.radio("Order", ['Desc', 'Asc'])
                
                filtered = df.copy()
                if search:
                    filtered = filtered[
                        filtered['name'].str.contains(search, case=False, na=False) |
                        filtered['lead_id'].astype(str).str.contains(search, case=False)
                    ]
                filtered = filtered[
                    (filtered['lead_score'] >= score_range[0]) &
                    (filtered['lead_score'] <= score_range[1])
                ]
                filtered = filtered.sort_values(sort_by, ascending=(sort_order == 'Asc'))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"📊 **{len(filtered):,}** of **{len(df):,}** leads")
                with col2:
                    if len(filtered) > 0:
                        st.success(f"⭐ Avg: **{filtered['lead_score'].mean():.1f}**")
                with col3:
                    if len(filtered) > 0:
                        hot_pct = (filtered['lead_category'] == 'Hot').sum() / len(filtered) * 100
                        st.warning(f"🔥 Hot: **{hot_pct:.1f}%**")
                
                st.dataframe(filtered, use_container_width=True, height=600)
            
            with tab5:
                st.markdown("### 💾 Export Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📄 Download CSV",
                        csv,
                        'scored_leads.csv',
                        'text/csv',
                        use_container_width=True
                    )
                
                with col2:
                    @st.cache_data
                    def to_excel(dataframe):
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            dataframe.to_excel(writer, index=False, sheet_name='Scored Leads')
                        return output.getvalue()
                    
                    excel = to_excel(df)
                    st.download_button(
                        "📊 Download Excel",
                        excel,
                        'scored_leads.xlsx',
                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                
                with col3:
                    hot_df = df[df['lead_category'] == 'Hot']
                    hot_csv = hot_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "🔥 Hot Leads Only",
                        hot_csv,
                        'hot_leads.csv',
                        'text/csv',
                        use_container_width=True
                    )
                
                st.markdown("---")
                st.markdown("#### Summary")
                
                summary = pd.DataFrame({
                    'Metric': [
                        '📊 Total Leads',
                        '🔥 Hot Leads',
                        '🌡️ Warm Leads',
                        '❄️ Cold Leads',
                        '⭐ Average Score',
                        '📈 Highest Score',
                        '📉 Lowest Score'
                    ],
                    'Value': [
                        f"{len(df):,}",
                        f"{len(df[df['lead_category'] == 'Hot']):,}",
                        f"{len(df[df['lead_category'] == 'Warm']):,}",
                        f"{len(df[df['lead_category'] == 'Cold']):,}",
                        f"{df['lead_score'].mean():.2f}",
                        f"{df['lead_score'].max()}",
                        f"{df['lead_score'].min()}"
                    ]
                })
                
                st.dataframe(summary, use_container_width=True, hide_index=True)
        
        else:
            st.markdown("""
                <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%); 
                            padding: 4rem 2rem; border-radius: 20px; text-align: center; border: 2px solid rgba(59, 130, 246, 0.3);'>
                    <h1 style='font-size: 3rem; margin-bottom: 1rem; color: #60a5fa;'>🚀 AI Lead Scoring</h1>
                    <p style='font-size: 1.3rem; color: #cbd5e1; opacity: 0.9;'>
                        Upload your CRM data and let AI analyze your leads
                    </p>
                    <p style='margin-top: 2rem; font-size: 1rem; color: #94a3b8;'>
                        👈 Select data source from the sidebar and click <b>TRAIN & SCORE</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # ADMIN TAB 2: USER MANAGEMENT
    # ========================================================================
    
    with admin_main_tab2:
        st.markdown("## 👑 User Management System")
        
        user_tab1, user_tab2, user_tab3, user_tab4 = st.tabs([
            "🟢 LIVE DASHBOARD",
            "➕ CREATE USER",
            "👥 MANAGE USERS",
            "📊 ACTIVITY LOG"
        ])
        
        with user_tab1:
            st.markdown("### 🟢 Live User Activity")
            
            if st.button("🔄 Refresh", key="admin_refresh"):
                st.rerun()
            
            active_users = get_currently_logged_in_users()
            
            if active_users:
                st.success(f"**{len(active_users)} user(s) online**")
                
                for user in active_users:
                    user_id, username, email, login_time, role = user
                    user_stats = get_user_stats(user_id)
                    
                    st.markdown(f"""
<div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
            padding: 20px; border-radius: 12px; margin: 12px 0; border: 2px solid rgba(16, 185, 129, 0.3);'>
    <span class='online-indicator'></span>
    <b style='color: #10b981; font-size: 1.1rem;'>{username}</b> 
    <span style='color: #6ee7b7; margin-left: 10px;'>({role})</span><br>
    <small style='color: #94a3b8;'>📧 {email if email else 'N/A'} | 🕒 {login_time}</small><br>
    <small style='color: #cbd5e1;'>📊 {user_stats['total_scorings']} scorings | 📄 {user_stats['total_leads']:,} leads</small>
</div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No users currently online")
        
        with user_tab2:
            st.markdown("### ➕ Create New User")
            
            with st.form("create_user_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    new_username = st.text_input("Username *")
                    new_email = st.text_input("Email")
                with col_b:
                    new_password = st.text_input("Password *", type="password")
                    confirm_password = st.text_input("Confirm Password *", type="password")
                
                submitted = st.form_submit_button("✅ CREATE USER", type="primary", use_container_width=True)
                
                if submitted:
                    if new_username and new_password == confirm_password and len(new_password) >= 6:
                        if create_user_by_admin(new_username, new_password, new_email):
                            st.success(f"""
✅ User Created Successfully!

**Username:** `{new_username}`
**Password:** `{new_password}`
**Email:** `{new_email if new_email else 'Not provided'}`

Please share these credentials with the user.
                            """)
                        else:
                            st.error("❌ Username already exists!")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        st.error("❌ Passwords don't match or fields are empty")
        
        with user_tab3:
            st.markdown("### 👥 All Users")
            
            users = get_all_users()
            user_data = []
            for user in users:
                user_data.append({
                    'ID': user[0],
                    'Username': user[1],
                    'Email': user[2] if user[2] else 'N/A',
                    'Created': user[3],
                    'Last Login': user[4] if user[4] else 'Never',
                    'Status': '🟢 Active' if user[5] else '🔴 Inactive',
                    'Role': user[6]
                })
            
            df_users = pd.DataFrame(user_data)
            st.dataframe(df_users, use_container_width=True, height=500)
            
            st.markdown("---")
            st.markdown("### ⚙️ User Actions")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                user_id_action = st.number_input("User ID", min_value=1, step=1)
            with col_m2:
                action_type = st.selectbox("Action", ["Enable", "Disable", "Delete"])
            with col_m3:
                st.write("")
                if st.button("▶️ EXECUTE", type="primary"):
                    if user_id_action != 1:  # Protect admin account
                        if action_type == "Enable":
                            toggle_user_status(user_id_action, 1)
                            st.success("✅ User enabled!")
                            time.sleep(1)
                            st.rerun()
                        elif action_type == "Disable":
                            toggle_user_status(user_id_action, 0)
                            st.warning("⚠️ User disabled!")
                            time.sleep(1)
                            st.rerun()
                        elif action_type == "Delete":
                            delete_user(user_id_action)
                            st.error("🗑️ User deleted!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("❌ Cannot modify admin account")
        
        with user_tab4:
            st.markdown("### 📊 System Activity Log")
            
            all_activities = get_all_user_activities()
            
            if all_activities:
                activity_data = []
                for activity in all_activities:
                    username, action, details, leads_scored, timestamp = activity
                    activity_data.append({
                        'Username': username,
                        'Action': action,
                        'Details': details if details else '-',
                        'Leads': leads_scored if leads_scored else '-',
                        'Timestamp': timestamp
                    })
                
                df_activities = pd.DataFrame(activity_data)
                st.dataframe(df_activities, use_container_width=True, height=600)
            else:
                st.info("No activities logged yet")

# ============================================================================
# USER DASHBOARD
# ============================================================================

else:
    
    st.markdown('<div class="main-header">🎯 LEAD SCORING DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Lead Intelligence & Analytics</div>', unsafe_allow_html=True)
    
    # Data source sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        upload_option = st.radio(
            "Select Source:",
            ["Use Default Dataset", "Upload Custom File"]
        )
        
        if upload_option == "Upload Custom File":
            uploaded_file = st.file_uploader(
                "Upload Excel File",
                type=['xlsx', 'xls']
            )
            data_path = uploaded_file
        else:
            data_path = "5000_rental_crm_leads.xlsx"
        
        st.markdown("---")
        
        train_button = st.button(
            "🚀 TRAIN & SCORE",
            type="primary",
            use_container_width=True
        )
    
    # Training
    if train_button and data_path:
        with st.spinner("🔄 Loading data..."):
            df = load_data(data_path)
        
        if df is not None:
            with st.expander("📊 Dataset Preview", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Rows", f"{len(df):,}")
                with col2:
                    st.metric("📋 Columns", len(df.columns))
                with col3:
                    st.metric("❓ Missing", df.isnull().sum().sum())
                with col4:
                    memory = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("💾 Memory", f"{memory:.2f} MB")
                
                st.dataframe(df.head(10), use_container_width=True)
            
            try:
                model, scored_df, features, accuracy, roc_auc = train_model(df)
                
                st.session_state['model'] = model
                st.session_state['scored_df'] = scored_df
                st.session_state['features'] = features
                st.session_state['accuracy'] = accuracy
                st.session_state['roc_auc'] = roc_auc
                
                log_usage(st.session_state.user['id'], 'score_leads', 'User scoring', len(scored_df))
                
                st.success("✅ Scoring complete!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Display results
    if 'scored_df' in st.session_state:
        df = st.session_state['scored_df']
        accuracy = st.session_state.get('accuracy', 0)
        roc_auc = st.session_state.get('roc_auc', None)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 DASHBOARD",
            "🔥 PRIORITY",
            "📈 ANALYTICS",
            "📋 ALL LEADS",
            "💾 EXPORT"
        ])
        
        with tab1:
            st.markdown("### 📊 Performance Dashboard")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📊 Total", f"{len(df):,}")
            with col2:
                hot = len(df[df['lead_category'] == 'Hot'])
                st.markdown(f"""
                    <div class="metric-card metric-card-hot">
                        <div style="font-size: 0.9rem; color: #fca5a5; font-weight: 700;">🔥 HOT</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: #ef4444; margin: 0.5rem 0;">{hot}</div>
                        <div style="font-size: 0.85rem; color: #fca5a5;">{hot/len(df)*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                warm = len(df[df['lead_category'] == 'Warm'])
                st.markdown(f"""
                    <div class="metric-card metric-card-warm">
                        <div style="font-size: 0.9rem; color: #fcd34d; font-weight: 700;">🌡️ WARM</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: #f59e0b; margin: 0.5rem 0;">{warm}</div>
                        <div style="font-size: 0.85rem; color: #fcd34d;">{warm/len(df)*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                cold = len(df[df['lead_category'] == 'Cold'])
                st.markdown(f"""
                    <div class="metric-card metric-card-cold">
                        <div style="font-size: 0.9rem; color: #93c5fd; font-weight: 700;">❄️ COLD</div>
                        <div style="font-size: 2.5rem; font-weight: 900; color: #3b82f6; margin: 0.5rem 0;">{cold}</div>
                        <div style="font-size: 0.85rem; color: #93c5fd;">{cold/len(df)*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col5:
                st.metric("⭐ Avg", f"{df['lead_score'].mean():.1f}")
            
            st.markdown("---")
            
            st.markdown("### 🎯 Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gauge = create_gauge_chart(accuracy*100, "Accuracy", "#3b82f6")
                st.plotly_chart(gauge, use_container_width=True)
            
            with col2:
                if roc_auc:
                    gauge = create_gauge_chart(roc_auc*100, "ROC AUC", "#8b5cf6")
                    st.plotly_chart(gauge, use_container_width=True)
                else:
                    st.info("ROC AUC not available")
            
            with col3:
                conversion = (df['lead_score'] > 70).sum() / len(df) * 100
                gauge = create_gauge_chart(conversion, "Hot %", "#10b981")
                st.plotly_chart(gauge, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                category_counts = df['lead_category'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#ef4444', '#f59e0b', '#3b82f6'])
                )])
                fig_pie.update_layout(
                    title="Categories",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter', 'color': '#e2e8f0'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df['lead_score'],
                    nbinsx=20,
                    marker=dict(color=df['lead_score'], colorscale='Viridis')
                ))
                fig_hist.update_layout(
                    title="Scores",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': 'Inter', 'color': '#e2e8f0'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔥 Priority Leads")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                category_filter = st.multiselect("Category", ['Hot', 'Warm', 'Cold'], default=['Hot'])
            with col2:
                min_score = st.slider("Min Score", 0, 100, 70)
            with col3:
                show_count = st.number_input("Show", 10, 100, 20, 10)
            
            filtered = df[
                (df['lead_category'].isin(category_filter)) & 
                (df['lead_score'] >= min_score)
            ]
            
            top_leads = filtered.nlargest(show_count, 'lead_score')
            st.dataframe(top_leads, use_container_width=True, height=600)
        
        with tab3:
            st.markdown("### 📈 Analytics")
            
            if 'source' in df.columns:
                source_stats = df.groupby('source')['lead_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=source_stats.index,
                    y=source_stats['mean'],
                    marker=dict(color=source_stats['mean'], colorscale='Viridis')
                ))
                fig_bar.update_layout(
                    title="Avg Score by Source",
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#e2e8f0'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab4:
            st.markdown("### 📋 All Leads")
            st.dataframe(df, use_container_width=True, height=600)
        
        with tab5:
            st.markdown("### 💾 Export")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 CSV", csv, 'leads.csv', use_container_width=True)
            
            with col2:
                @st.cache_data
                def to_excel(dataframe):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        dataframe.to_excel(writer, index=False)
                    return output.getvalue()
                
                excel = to_excel(df)
                st.download_button("📊 Excel", excel, 'leads.xlsx', use_container_width=True)
            
            with col3:
                hot_csv = df[df['lead_category'] == 'Hot'].to_csv(index=False).encode('utf-8')
                st.download_button("🔥 Hot Only", hot_csv, 'hot_leads.csv', use_container_width=True)
    
    else:
        st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%); 
                        padding: 4rem 2rem; border-radius: 20px; text-align: center; border: 2px solid rgba(59, 130, 246, 0.3);'>
                <h1 style='font-size: 3rem; color: #60a5fa;'>🚀 Get Started</h1>
                <p style='font-size: 1.3rem; color: #cbd5e1; margin-top: 1rem;'>
                    Upload your CRM data and score your leads with AI
                </p>
                <p style='margin-top: 2rem; color: #94a3b8;'>
                    👈 Select data source and click <b>TRAIN & SCORE</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #94a3b8; padding: 1rem 0;'>
    <p>🔐 Logged in as: <b style='color: #60a5fa;'>{st.session_state.user['username']}</b> 
       ({st.session_state.user['role']}) | 
       {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p style='opacity: 0.7;'>✨ AI Lead Scoring Pro | Version 2.0 | © 2024</p>
</div>
""", unsafe_allow_html=True)