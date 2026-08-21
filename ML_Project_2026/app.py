import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CLV Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ================= GLOBAL ================= */

    .stApp {
        background-color: #f4f7fb;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #172033;
    }

    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #172554 55%,
            #1e1b4b 100%
        );
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    /* ================= BRAND ================= */

    .brand {
        text-align: center;
        padding: 10px 5px 20px 5px;
    }

    .brand-icon {
        font-size: 48px;
    }

    .brand-title {
        font-size: 21px;
        font-weight: 800;
        color: white;
        margin-top: 5px;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #a5b4fc;
        margin-top: 4px;
    }

    .sidebar-bottom {
        margin-top: 30px;
        padding: 15px;
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        text-align: center;
        font-size: 12px;
        color: #c7d2fe;
    }

    /* ================= HERO ================= */

    .hero {
        background: linear-gradient(
            135deg,
            #2563eb 0%,
            #4f46e5 50%,
            #7c3aed 100%
        );

        border-radius: 22px;
        padding: 34px 38px;
        color: white;
        box-shadow: 0 12px 35px rgba(79,70,229,0.25);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-text {
        font-size: 16px;
        color: #e0e7ff;
        max-width: 800px;
        line-height: 1.6;
    }

    .online {
        float: right;
        background: rgba(255,255,255,0.15);
        padding: 8px 14px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 600;
    }

    /* ================= KPI ================= */

    .kpi-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 22px;
        min-height: 145px;
        box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    }

    .kpi-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .kpi-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 29px;
        font-weight: 800;
        color: #111827;
        margin-top: 7px;
    }

    .kpi-small {
        font-size: 12px;
        color: #10b981;
        margin-top: 5px;
    }

    /* ================= SECTION ================= */

    .section {
        font-size: 22px;
        font-weight: 800;
        color: #172033;
        margin-top: 28px;
        margin-bottom: 16px;
    }

    /* ================= FEATURE CARD ================= */

    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 24px;
        min-height: 180px;
        box-shadow: 0 5px 18px rgba(15,23,42,0.04);
    }

    .feature-icon {
        font-size: 35px;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 18px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 8px;
    }

    .feature-text {
        font-size: 14px;
        color: #64748b;
        line-height: 1.6;
    }

    /* ================= INPUT CARD ================= */

    .input-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    }

    /* ================= PREDICTION ================= */

    .prediction-card {
        background: linear-gradient(
            135deg,
            #ecfdf5,
            #d1fae5
        );

        border: 1px solid #86efac;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 15px;
    }

    .prediction-title {
        color: #166534;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .prediction-value {
        color: #15803d;
        font-size: 42px;
        font-weight: 900;
        margin: 8px 0;
    }

    .prediction-status {
        display: inline-block;
        background: #16a34a;
        color: white;
        padding: 7px 16px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 700;
    }

    /* ================= SEGMENT CARD ================= */

    .segment-card {
        background: white;
        border-radius: 17px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15,23,42,0.04);
    }

    .segment-number {
        font-size: 28px;
        font-weight: 800;
        color: #4f46e5;
    }

    .segment-label {
        color: #64748b;
        font-size: 13px;
    }

    /* ================= ADMIN ================= */

    .login-card {
        background: white;
        border-radius: 20px;
        padding: 35px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 30px rgba(15,23,42,0.07);
        max-width: 550px;
        margin: auto;
    }

    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding-top: 35px;
    }

    /* ================= BUTTON ================= */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    /* ================= FILE UPLOADER ================= */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "user_inputs.xlsx"


# ============================================================
# SESSION STATE
# ============================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "attempts" not in st.session_state:
    st.session_state.attempts = 0

if "blocked" not in st.session_state:
    st.session_state.blocked = False


# ============================================================
# MODEL
# ============================================================

def train_clv_model():

    data = {
        "Recency": [
            10, 20, 5, 30, 15, 40,
            25, 8, 60, 12, 35, 18
        ],

        "Frequency": [
            5, 3, 10, 2, 7, 1,
            4, 12, 2, 8, 3, 6
        ],

        "Monetary": [
            500, 300, 1000, 200,
            700, 100, 400, 1500,
            250, 900, 350, 650
        ],

        "CLV": [
            1200, 700, 2500, 400,
            1600, 200, 900, 3000,
            500, 2000, 800, 1400
        ]
    }

    df = pd.DataFrame(data)

    X = df[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ]

    y = df["CLV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = GradientBoostingRegressor(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model, df


# ============================================================
# SAVE DATA
# ============================================================

def save_user_data(data):

    if os.path.exists(DATA_FILE):

        try:

            old_data = pd.read_excel(
                DATA_FILE
            )

            final_data = pd.concat(
                [
                    old_data,
                    data
                ],
                ignore_index=True
            )

            final_data.to_excel(
                DATA_FILE,
                index=False
            )

        except:

            data.to_excel(
                DATA_FILE,
                index=False
            )

    else:

        data.to_excel(
            DATA_FILE,
            index=False
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="brand">

        <div class="brand-icon">
            📊
        </div>

        <div class="brand-title">
            CLV INTELLIGENCE
        </div>

        <div class="brand-subtitle">
            Customer Analytics Platform
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        "**MAIN MENU**"
    )

    menu = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 CLV Predictor",
            "👥 Customer Segments",
            "🔐 Admin Panel"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div class="sidebar-bottom">

        🤖 Machine Learning<br>
        📊 Customer Analytics<br>
        🔐 Secure Data Management

        <br><br>

        <b>System Status</b><br>
        🟢 Online

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if menu == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">

        <div class="online">
            ● System Online
        </div>

        <div class="hero-title">
            Welcome to CLV Intelligence 👋
        </div>

        <div class="hero-text">
            AI-powered Customer Lifetime Value analytics
            platform for predicting customer value,
            understanding behaviour and creating meaningful
            customer segments.
        </div>

    </div>
    """, unsafe_allow_html=True)

    # ---------------- DATA ----------------

    customer_count = 12
    avg_clv = 1333
    max_clv = 3000

    if os.path.exists(DATA_FILE):

        try:

            stored = pd.read_excel(
                DATA_FILE
            )

            if not stored.empty:

                customer_count = len(stored)

                if "Predicted_CLV" in stored.columns:

                    avg_clv = stored[
                        "Predicted_CLV"
                    ].mean()

                    max_clv = stored[
                        "Predicted_CLV"
                    ].max()

        except:
            pass

    # ---------------- KPI ----------------

    st.markdown(
        '<div class="section">📊 Business Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(f"""
        <div class="kpi-card">

            <div class="kpi-icon">👥</div>

            <div class="kpi-label">
                CUSTOMERS ANALYZED
            </div>

            <div class="kpi-value">
                {customer_count}
            </div>

            <div class="kpi-small">
                Customer records
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="kpi-card">

            <div class="kpi-icon">💰</div>

            <div class="kpi-label">
                AVERAGE CLV
            </div>

            <div class="kpi-value">
                ${avg_clv:,.0f}
            </div>

            <div class="kpi-small">
                Estimated value
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class="kpi-card">

            <div class="kpi-icon">🏆</div>

            <div class="kpi-label">
                HIGHEST CLV
            </div>

            <div class="kpi-value">
                ${max_clv:,.0f}
            </div>

            <div class="kpi-small">
                Top customer value
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c4:

        st.markdown("""
        <div class="kpi-card">

            <div class="kpi-icon">🤖</div>

            <div class="kpi-label">
                ML MODEL
            </div>

            <div class="kpi-value">
                ACTIVE
            </div>

            <div class="kpi-small">
                Gradient Boosting
            </div>

        </div>
        """, unsafe_allow_html=True)

    # ---------------- FEATURES ----------------

    st.markdown(
        '<div class="section">🚀 Platform Capabilities</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                🔮
            </div>

            <div class="feature-title">
                CLV Prediction
            </div>

            <div class="feature-text">
                Predict future customer lifetime value
                using Recency, Frequency and Monetary
                behaviour with a machine learning model.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                👥
            </div>

            <div class="feature-title">
                Customer Segmentation
            </div>

            <div class="feature-text">
                Automatically divide customers into
                meaningful groups using K-Means clustering.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="feature-card">

            <div class="feature-icon">
                🔐
            </div>

            <div class="feature-title">
                Secure Administration
            </div>

            <div class="feature-text">
                Manage prediction records and download
                stored customer information through
                the protected admin panel.
            </div>

        </div>
        """, unsafe_allow_html=True)

    # ---------------- WORKFLOW ----------------

    st.markdown(
        '<div class="section">⚙️ How the System Works</div>',
        unsafe_allow_html=True
    )

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.info(
            "### 01\n"
            "📥 **Input Data**\n\n"
            "Enter customer behaviour."
        )

    with w2:
        st.info(
            "### 02\n"
            "🧠 **ML Processing**\n\n"
            "Gradient Boosting analyzes data."
        )

    with w3:
        st.info(
            "### 03\n"
            "💰 **CLV Prediction**\n\n"
            "Calculate estimated customer value."
        )

    with w4:
        st.info(
            "### 04\n"
            "📊 **Analytics**\n\n"
            "Understand customer segments."
        )


# ============================================================
# CLV PREDICTOR
# ============================================================

elif menu == "🔮 CLV Predictor":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            🔮 Customer Lifetime Value Predictor
        </div>

        <div class="hero-text">
            Enter customer purchase behaviour and
            use machine learning to estimate their
            future lifetime value.
        </div>

    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    # ========================================================
    # INPUT
    # ========================================================

    with left:

        st.markdown(
            '<div class="section">📋 Customer Information</div>',
            unsafe_allow_html=True
        )

        recency = st.number_input(
            "📅 Recency",
            min_value=0,
            max_value=365,
            value=30,
            step=1
        )

        st.caption(
            "Days since the customer's last purchase."
        )

        frequency = st.number_input(
            "🔄 Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )

        st.caption(
            "Number of purchases made by the customer."
        )

        monetary = st.number_input(
            "💰 Monetary Value",
            min_value=0.0,
            max_value=100000.0,
            value=500.0,
            step=50.0
        )

        st.caption(
            "Total amount spent by the customer."
        )

        st.write("")

        predict = st.button(
            "🚀 Predict Customer CLV",
            type="primary",
            use_container_width=True
        )

    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    with right:

        st.markdown(
            '<div class="section">👤 Customer Profile</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="input-card">',
            unsafe_allow_html=True
        )

        p1, p2 = st.columns(2)

        with p1:

            st.metric(
                "📅 Recency",
                f"{recency} days"
            )

        with p2:

            st.metric(
                "🔄 Frequency",
                frequency
            )

        st.metric(
            "💰 Monetary Value",
            f"${monetary:,.2f}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict:

        model, training_data = train_clv_model()

        user_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = float(
            model.predict(
                user_data
            )[0]
        )

        user_data[
            "Predicted_CLV"
        ] = prediction

        save_user_data(
            user_data
        )

        # Determine value level

        if prediction >= 2000:

            level = "HIGH VALUE"

        elif prediction >= 1000:

            level = "MEDIUM VALUE"

        else:

            level = "STANDARD VALUE"

        st.markdown(
            '<div class="section">💎 Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="prediction-card">

            <div class="prediction-title">
                ESTIMATED CUSTOMER LIFETIME VALUE
            </div>

            <div class="prediction-value">
                ${prediction:,.2f}
            </div>

            <div class="prediction-status">
                {level}
            </div>

        </div>
        """, unsafe_allow_html=True)

        # ====================================================
        # CHART
        # ====================================================

        st.markdown(
            '<div class="section">📈 Customer Behaviour</div>',
            unsafe_allow_html=True
        )

        chart_df = pd.DataFrame({
            "Metric": [
                "Recency",
                "Frequency",
                "Monetary"
            ],

            "Value": [
                recency,
                frequency,
                monetary
            ]
        })

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        ax.bar(
            chart_df["Metric"],
            chart_df["Value"]
        )

        ax.set_title(
            "Customer Behaviour Profile",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_ylabel(
            "Value"
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        st.success(
            "✅ Prediction completed and saved successfully."
        )


# ============================================================
# SEGMENTATION
# ============================================================

elif menu == "👥 Customer Segments":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            👥 Customer Segmentation
        </div>

        <div class="hero-text">
            Discover customer behaviour patterns and
            divide customers into meaningful groups
            using K-Means clustering.
        </div>

    </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # UPLOAD
    # ========================================================

    st.markdown(
        '<div class="section">📂 Upload Dataset</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload your customer CSV file",
        type=["csv"]
    )

    if uploaded_file:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"Dataset loaded successfully — {len(df)} records."
            )

            st.markdown(
                '<div class="section">👀 Dataset Preview</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.markdown(
                '<div class="section">⚙️ Configure Segmentation</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                r = st.selectbox(
                    "📅 Recency Column",
                    df.columns
                )

            with c2:

                f = st.selectbox(
                    "🔄 Frequency Column",
                    df.columns
                )

            with c3:

                m = st.selectbox(
                    "💰 Monetary Column",
                    df.columns
                )

            clusters = st.slider(
                "🎯 Number of Customer Segments",
                min_value=2,
                max_value=6,
                value=3
            )

            st.write("")

            run = st.button(
                "🚀 Run Customer Segmentation",
                type="primary",
                use_container_width=True
            )

            if run:

                X = df[
                    [r, f, m]
                ].copy()

                X = X.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                valid = X.notna().all(
                    axis=1
                )

                df = df.loc[
                    valid
                ].copy()

                X = X.loc[
                    valid
                ]

                if len(X) < clusters:

                    st.error(
                        "Not enough valid records for the selected number of clusters."
                    )

                else:

                    scaler = StandardScaler()

                    X_scaled = scaler.fit_transform(
                        X
                    )

                    kmeans = KMeans(
                        n_clusters=clusters,
                        random_state=42,
                        n_init=10
                    )

                    df["Cluster"] = kmeans.fit_predict(
                        X_scaled
                    )

                    st.success(
                        "✅ Segmentation completed successfully!"
                    )

                    # =================================================
                    # SEGMENT KPIs
                    # =================================================

                    st.markdown(
                        '<div class="section">📊 Segmentation Overview</div>',
                        unsafe_allow_html=True
                    )

                    s1, s2, s3, s4 = st.columns(4)

                    with s1:

                        st.markdown(f"""
                        <div class="kpi-card">

                            <div class="kpi-icon">👥</div>

                            <div class="kpi-label">
                                CUSTOMERS
                            </div>

                            <div class="kpi-value">
                                {len(df)}
                            </div>

                        </div>
                        """, unsafe_allow_html=True)

                    with s2:

                        st.markdown(f"""
                        <div class="kpi-card">

                            <div class="kpi-icon">🎯</div>

                            <div class="kpi-label">
                                SEGMENTS
                            </div>

                            <div class="kpi-value">
                                {clusters}
                            </div>

                        </div>
                        """, unsafe_allow_html=True)

                    with s3:

                        st.markdown(f"""
                        <div class="kpi-card">

                            <div class="kpi-icon">💰</div>

                            <div class="kpi-label">
                                AVG MONETARY
                            </div>

                            <div class="kpi-value">
                                ${X[m].mean():,.0f}
                            </div>

                        </div>
                        """, unsafe_allow_html=True)

                    with s4:

                        st.markdown(f"""
                        <div class="kpi-card">

                            <div class="kpi-icon">📈</div>

                            <div class="kpi-label">
                                DATA ROWS
                            </div>

                            <div class="kpi-value">
                                {len(X)}
                            </div>

                        </div>
                        """, unsafe_allow_html=True)

                    # =================================================
                    # CHART
                    # =================================================

                    st.markdown(
                        '<div class="section">📈 Customer Segmentation Map</div>',
                        unsafe_allow_html=True
                    )

                    fig, ax = plt.subplots(
                        figsize=(11, 5)
                    )

                    scatter = ax.scatter(
                        df[r],
                        df[m],
                        c=df["Cluster"],
                        cmap="viridis",
                        s=85,
                        alpha=0.8
                    )

                    ax.set_xlabel(
                        "Recency"
                    )

                    ax.set_ylabel(
                        "Monetary Value"
                    )

                    ax.set_title(
                        "Customer Segmentation",
                        fontsize=15,
                        fontweight="bold"
                    )

                    ax.grid(
                        alpha=0.2
                    )

                    plt.colorbar(
                        scatter,
                        ax=ax,
                        label="Cluster"
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    # =================================================
                    # SUMMARY
                    # =================================================

                    st.markdown(
                        '<div class="section">📋 Segment Summary</div>',
                        unsafe_allow_html=True
                    )

                    summary = df.groupby(
                        "Cluster"
                    )[
                        [r, f, m]
                    ].mean().round(2)

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )

                    # =================================================
                    # FULL DATA
                    # =================================================

                    st.markdown(
                        '<div class="section">👥 Customer Classification</div>',
                        unsafe_allow_html=True
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    csv_data = df.to_csv(
                        index=False
                    )

                    st.download_button(
                        "📥 Download Segmented Dataset",
                        csv_data,
                        "segmented_customers.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Unable to process dataset: {e}"
            )

    else:

        st.info(
            "📂 Upload a CSV dataset above to begin customer segmentation."
        )


# ============================================================
# ADMIN PANEL
# ============================================================

elif menu == "🔐 Admin Panel":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            🔐 Administration Center
        </div>

        <div class="hero-text">
            Securely manage stored customer predictions,
            inspect application files and download analytics.
        </div>

    </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # BLOCKED
    # ========================================================

    if st.session_state.blocked:

        st.error(
            "🚫 Admin access has been blocked after 3 incorrect attempts."
        )

        st.warning(
            "Please contact the administrator to request access."
        )

        st.link_button(
            "📧 Contact Administrator",
            "https://mail.google.com/mail/?view=cm&fs=1&to=atharavshende999@gmail.com&su=CLV%20Admin%20Access%20Request",
            use_container_width=True
        )

        st.stop()

    # ========================================================
    # LOGGED IN
    # ========================================================

    if st.session_state.admin_logged_in:

        st.success(
            "🟢 Administrator authenticated successfully."
        )

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.admin_logged_in = False

            st.rerun()

        # ====================================================
        # STORED DATA
        # ====================================================

        st.markdown(
            '<div class="section">📊 Customer Prediction Records</div>',
            unsafe_allow_html=True
        )

        if os.path.exists(DATA_FILE):

            try:

                df = pd.read_excel(
                    DATA_FILE
                )

                if not df.empty:

                    a, b, c = st.columns(3)

                    with a:

                        st.metric(
                            "👥 Total Records",
                            len(df)
                        )

                    with b:

                        if "Predicted_CLV" in df.columns:

                            st.metric(
                                "💰 Average CLV",
                                f"${df['Predicted_CLV'].mean():,.2f}"
                            )

                    with c:

                        if "Predicted_CLV" in df.columns:

                            st.metric(
                                "🏆 Highest CLV",
                                f"${df['Predicted_CLV'].max():,.2f}"
                            )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    with open(
                        DATA_FILE,
                        "rb"
                    ) as file:

                        st.download_button(
                            "📥 Download Excel Data",
                            file,
                            "user_inputs.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                else:

                    st.warning(
                        "No customer records available."
                    )

            except Exception as e:

                st.error(
                    f"Could not read Excel file: {e}"
                )

        else:

            st.info(
                "📭 No prediction data exists yet."
            )

        # ====================================================
        # SERVER FILES
        # ====================================================

        st.markdown(
            '<div class="section">📂 Application Files</div>',
            unsafe_allow_html=True
        )

        with st.expander(
            "View Application Files"
        ):

            files = os.listdir()

            for file in files:

                st.write(
                    f"📄 {file}"
                )

    # ========================================================
    # LOGIN
    # ========================================================

    else:

        st.markdown("""
        <div class="login-card">

            <div style="
                text-align:center;
                font-size:55px;
            ">
                🔐
            </div>

            <h2 style="text-align:center;">
                Administrator Login
            </h2>

            <p style="
                text-align:center;
                color:#64748b;
            ">
                Authorized users only
            </p>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        password = st.text_input(
            "🔑 Administrator Password",
            type="password",
            placeholder="Enter password"
        )

        login = st.button(
            "🔐 Secure Login",
            type="primary",
            use_container_width=True
        )

        if login:

            if password.strip() == "admin123":

                st.session_state.attempts = 0

                st.session_state.admin_logged_in = True

                st.success(
                    "✅ Access granted."
                )

                st.rerun()

            else:

                st.session_state.attempts += 1

                remaining = (
                    3 -
                    st.session_state.attempts
                )

                if remaining > 0:

                    st.error(
                        f"❌ Incorrect password. "
                        f"{remaining} attempt(s) remaining."
                    )

                else:

                    st.session_state.blocked = True

                    st.error(
                        "🚫 Access blocked."
                    )

                    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <b>CLV Intelligence Dashboard</b>

    <br>

    Customer Lifetime Value • Machine Learning •
    Customer Segmentation

    <br><br>

    Built with Python • Streamlit • Pandas • Scikit-Learn

</div>
""", unsafe_allow_html=True)
