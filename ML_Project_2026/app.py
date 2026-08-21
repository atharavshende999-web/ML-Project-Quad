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
    page_title="CLV Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background: #f5f7fb;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding: 10px;
        border-radius: 8px;
    }

    /* ---------- HEADINGS ---------- */

    h1 {
        color: #111827;
        font-weight: 800;
    }

    h2 {
        color: #1e293b;
        font-weight: 700;
    }

    h3 {
        color: #334155;
    }

    /* ---------- CARDS ---------- */

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-icon {
        font-size: 28px;
        margin-bottom: 5px;
    }

    /* ---------- HERO ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #2563eb 0%,
            #4f46e5 100%
        );
        padding: 35px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
    }

    .hero h1 {
        color: white;
        font-size: 36px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 16px;
        margin: 0;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    }

    .info-card h3 {
        margin-top: 0;
        color: #1e293b;
    }

    .info-card p {
        color: #64748b;
        line-height: 1.6;
    }

    /* ---------- PREDICTION RESULT ---------- */

    .prediction-box {
        background: linear-gradient(
            135deg,
            #ecfdf5,
            #d1fae5
        );
        border: 1px solid #86efac;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        color: #166534;
        font-size: 15px;
        font-weight: 600;
    }

    .prediction-value {
        color: #15803d;
        font-size: 40px;
        font-weight: 800;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #1e293b;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 700;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE CONFIG
# ============================================================

DATA_FILE = "user_inputs.xlsx"


# ============================================================
# MODEL FUNCTION
# ============================================================

def train_clv_model():

    data = {
        "Recency": [10, 20, 5, 30, 15, 40, 25, 8, 60, 12, 35, 18],
        "Frequency": [5, 3, 10, 2, 7, 1, 4, 12, 2, 8, 3, 6],
        "Monetary": [500, 300, 1000, 200, 700, 100, 400,
                     1500, 250, 900, 350, 650],
        "CLV": [1200, 700, 2500, 400, 1600, 200, 900,
                3000, 500, 2000, 800, 1400]
    }

    df = pd.DataFrame(data)

    X = df[["Recency", "Frequency", "Monetary"]]
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

    model.fit(X_train, y_train)

    return model, df


# ============================================================
# SAVE USER DATA
# ============================================================

def save_user_data(user_data):

    if os.path.exists(DATA_FILE):

        try:
            old_data = pd.read_excel(DATA_FILE)

            new_data = pd.concat(
                [old_data, user_data],
                ignore_index=True
            )

            new_data.to_excel(
                DATA_FILE,
                index=False
            )

        except Exception:
            user_data.to_excel(
                DATA_FILE,
                index=False
            )

    else:
        user_data.to_excel(
            DATA_FILE,
            index=False
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="
        text-align:center;
        padding:15px 5px 25px 5px;
    ">
        <div style="font-size:45px;">📊</div>
        <h2 style="margin:0;color:white;">
            CLV Intelligence
        </h2>
        <p style="
            color:#94a3b8;
            font-size:13px;
        ">
            Customer Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    menu = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🔮 CLV Predictor",
            "📊 Segmentation Dashboard",
            "🔐 Admin Panel"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.caption("🤖 Machine Learning")
    st.caption("📈 Customer Analytics")
    st.caption("🔐 Secure Administration")


# ============================================================
# HOME
# ============================================================

if menu == "🏠 Home":

    st.markdown("""
    <div class="hero">

        <h1>📊 Customer Lifetime Value</h1>

        <p>
        AI-powered customer analytics dashboard for
        predicting CLV and identifying valuable customer segments.
        </p>

    </div>
    """, unsafe_allow_html=True)

    # KPI CARDS

    total_customers = 12
    avg_clv = 1333
    max_clv = 3000

    if os.path.exists(DATA_FILE):

        try:
            stored = pd.read_excel(DATA_FILE)

            if len(stored) > 0:

                total_customers = len(stored)

                if "Predicted_CLV" in stored.columns:
                    avg_clv = stored["Predicted_CLV"].mean()
                    max_clv = stored["Predicted_CLV"].max()

        except:
            pass

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-icon">👥</div>

            <div class="metric-title">
                Customers Analyzed
            </div>

            <div class="metric-value">
                {total_customers}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-icon">💰</div>

            <div class="metric-title">
                Average CLV
            </div>

            <div class="metric-value">
                ${avg_clv:,.0f}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">

            <div class="metric-icon">🏆</div>

            <div class="metric-title">
                Highest CLV
            </div>

            <div class="metric-value">
                ${max_clv:,.0f}
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown("""
        <div class="metric-card">

            <div class="metric-icon">🤖</div>

            <div class="metric-title">
                ML Model
            </div>

            <div class="metric-value">
                Active
            </div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🚀 Platform Features</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="info-card">

            <h3>🔮 CLV Prediction</h3>

            <p>
            Predict the potential lifetime value of a customer
            using Recency, Frequency and Monetary behaviour.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="info-card">

            <h3>👥 Customer Segmentation</h3>

            <p>
            Use K-Means clustering to divide customers into
            meaningful behavioural groups.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="info-card">

            <h3>🔐 Admin Analytics</h3>

            <p>
            Securely access stored prediction records and
            download customer analytics data.
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🧠 How CLV Works</div>',
        unsafe_allow_html=True
    )

    step1, step2, step3 = st.columns(3)

    with step1:

        st.info(
            "1️⃣ **Customer Data**\n\n"
            "Recency • Frequency • Monetary"
        )

    with step2:

        st.info(
            "2️⃣ **Machine Learning**\n\n"
            "Gradient Boosting Regression"
        )

    with step3:

        st.success(
            "3️⃣ **CLV Result**\n\n"
            "Estimated customer value"
        )


# ============================================================
# CLV PREDICTOR
# ============================================================

elif menu == "🔮 CLV Predictor":

    st.markdown("""
    <div class="hero">

        <h1>🔮 CLV Predictor</h1>

        <p>
        Enter customer behaviour data to estimate
        their Customer Lifetime Value.
        </p>

    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:

        st.markdown(
            '<div class="section-title">📋 Customer Information</div>',
            unsafe_allow_html=True
        )

        recency = st.number_input(
            "📅 Recency (Days)",
            min_value=0,
            max_value=365,
            value=30,
            help="Number of days since the customer's last purchase."
        )

        frequency = st.number_input(
            "🔄 Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5,
            help="Number of purchases made by the customer."
        )

        monetary = st.number_input(
            "💵 Monetary Value ($)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=50.0,
            help="Total monetary value generated by the customer."
        )

        predict_button = st.button(
            "🚀 Predict Customer CLV",
            use_container_width=True,
            type="primary"
        )

    with right:

        st.markdown(
            '<div class="section-title">📌 Customer Profile</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Recency",
            f"{recency} days"
        )

        st.metric(
            "Purchase Frequency",
            f"{frequency}"
        )

        st.metric(
            "Monetary Value",
            f"${monetary:,.2f}"
        )

    if predict_button:

        model, training_data = train_clv_model()

        user_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = model.predict(user_data)[0]

        user_data["Predicted_CLV"] = prediction

        save_user_data(user_data)

        st.markdown(f"""
        <div class="prediction-box">

            <div class="prediction-label">
                ESTIMATED CUSTOMER LIFETIME VALUE
            </div>

            <div class="prediction-value">
                ${prediction:,.2f}
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Prediction Analysis")

        chart_data = pd.DataFrame({
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

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            chart_data["Metric"],
            chart_data["Value"]
        )

        ax.set_title(
            "Customer Behaviour Profile"
        )

        ax.set_ylabel("Value")

        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(fig)

        st.success(
            "✅ Prediction completed and customer data saved successfully."
        )


# ============================================================
# SEGMENTATION DASHBOARD
# ============================================================

elif menu == "📊 Segmentation Dashboard":

    st.markdown("""
    <div class="hero">

        <h1>📊 Customer Segmentation</h1>

        <p>
        Discover customer groups using K-Means clustering.
        </p>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload Customer CSV Dataset",
        type=["csv"]
    )

    if uploaded_file:

        try:

            df = pd.read_csv(uploaded_file)

            st.success(
                f"✅ Dataset loaded successfully — {len(df)} records found."
            )

            st.markdown(
                '<div class="section-title">👀 Dataset Preview</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.markdown(
                '<div class="section-title">⚙️ Select Features</div>',
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
                "Number of Customer Segments",
                min_value=2,
                max_value=6,
                value=3
            )

            if st.button(
                "🚀 Run Customer Segmentation",
                use_container_width=True,
                type="primary"
            ):

                X = df[[r, f, m]].copy()

                # Convert values to numeric
                X = X.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                # Remove invalid rows
                valid_rows = X.notna().all(axis=1)

                df = df.loc[valid_rows].copy()

                X = X.loc[valid_rows]

                if len(X) < clusters:

                    st.error(
                        "Not enough valid records for the selected number of clusters."
                    )

                else:

                    scaler = StandardScaler()

                    X_scaled = scaler.fit_transform(X)

                    kmeans = KMeans(
                        n_clusters=clusters,
                        random_state=42,
                        n_init=10
                    )

                    df["Cluster"] = kmeans.fit_predict(
                        X_scaled
                    )

                    st.success(
                        "✅ Customer segmentation completed!"
                    )

                    # ---------- KPI ----------

                    a, b, c = st.columns(3)

                    with a:

                        st.metric(
                            "👥 Customers",
                            len(df)
                        )

                    with b:

                        st.metric(
                            "🎯 Segments",
                            clusters
                        )

                    with c:

                        st.metric(
                            "💰 Avg Monetary",
                            f"${X[m].mean():,.0f}"
                        )

                    # ---------- CHART ----------

                    st.markdown(
                        '<div class="section-title">📈 Customer Segments</div>',
                        unsafe_allow_html=True
                    )

                    fig, ax = plt.subplots(
                        figsize=(10, 5)
                    )

                    scatter = ax.scatter(
                        df[r],
                        df[m],
                        c=df["Cluster"],
                        cmap="viridis",
                        s=80,
                        alpha=0.8
                    )

                    ax.set_xlabel(
                        "Recency"
                    )

                    ax.set_ylabel(
                        "Monetary Value"
                    )

                    ax.set_title(
                        "Customer Segmentation Map"
                    )

                    ax.grid(
                        alpha=0.2
                    )

                    plt.colorbar(
                        scatter,
                        ax=ax,
                        label="Cluster"
                    )

                    st.pyplot(fig)

                    # ---------- CLUSTER SUMMARY ----------

                    st.markdown(
                        '<div class="section-title">📋 Segment Summary</div>',
                        unsafe_allow_html=True
                    )

                    summary = df.groupby(
                        "Cluster"
                    )[[r, f, m]].mean().round(2)

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )

                    # ---------- FULL DATA ----------

                    st.markdown(
                        '<div class="section-title">👥 Classified Customers</div>',
                        unsafe_allow_html=True
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"❌ Unable to process dataset: {e}"
            )

    else:

        st.info(
            "📂 Upload a CSV file above to start customer segmentation."
        )


# ============================================================
# ADMIN PANEL
# ============================================================

elif menu == "🔐 Admin Panel":

    st.markdown("""
    <div class="hero">

        <h1>🔐 Admin Panel</h1>

        <p>
        Secure access to stored customer prediction data.
        </p>

    </div>
    """, unsafe_allow_html=True)

    # Session state

    if "attempts" not in st.session_state:
        st.session_state.attempts = 0

    if "blocked" not in st.session_state:
        st.session_state.blocked = False

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    # ========================================================
    # BLOCKED
    # ========================================================

    if st.session_state.blocked:

        st.error(
            "🚫 Too many incorrect attempts. "
            "Admin access has been temporarily blocked."
        )

        st.markdown(
            "### 📩 Need administrator access?"
        )

        email = "atharavshende999@gmail.com"

        gmail_link = (
            "https://mail.google.com/mail/"
            "?view=cm&fs=1"
            f"&to={email}"
            "&su=Access Request for CLV App"
        )

        st.link_button(
            "📧 Contact Admin",
            gmail_link,
            use_container_width=True
        )

        st.stop()

    # ========================================================
    # ALREADY LOGGED IN
    # ========================================================

    if st.session_state.admin_logged_in:

        st.success(
            "✅ Admin authentication successful."
        )

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.admin_logged_in = False

            st.rerun()

        st.markdown(
            '<div class="section-title">📂 Server Files</div>',
            unsafe_allow_html=True
        )

        files = os.listdir()

        st.info(
            f"📁 {len(files)} files found in application directory."
        )

        with st.expander(
            "View Server Files"
        ):

            for file in files:

                st.write(
                    f"📄 {file}"
                )

        # ====================================================
        # USER DATA
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Stored Customer Predictions</div>',
            unsafe_allow_html=True
        )

        if os.path.exists(DATA_FILE):

            try:

                df = pd.read_excel(
                    DATA_FILE
                )

                if len(df) > 0:

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "👥 Records",
                            len(df)
                        )

                    with c2:

                        if "Predicted_CLV" in df.columns:

                            st.metric(
                                "💰 Average CLV",
                                f"${df['Predicted_CLV'].mean():,.2f}"
                            )

                    with c3:

                        if "Predicted_CLV" in df.columns:

                            st.metric(
                                "🏆 Highest CLV",
                                f"${df['Predicted_CLV'].max():,.2f}"
                            )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    # Download

                    with open(
                        DATA_FILE,
                        "rb"
                    ) as file:

                        st.download_button(
                            label="📥 Download Customer Data",
                            data=file,
                            file_name="user_inputs.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            ),
                            use_container_width=True
                        )

                else:

                    st.warning(
                        "No customer records found."
                    )

            except Exception as e:

                st.error(
                    f"Unable to read data: {e}"
                )

        else:

            st.warning(
                "📭 No prediction data has been stored yet."
            )

    # ========================================================
    # LOGIN
    # ========================================================

    else:

        st.markdown("""
        <div style="
            max-width:500px;
            margin:auto;
            background:white;
            padding:30px;
            border-radius:18px;
            box-shadow:0 5px 20px rgba(0,0,0,0.07);
        ">

        <h2 style="text-align:center;">
        🔑 Administrator Login
        </h2>

        <p style="
            text-align:center;
            color:#64748b;
        ">
        Enter the administrator password to continue.
        </p>

        </div>
        """, unsafe_allow_html=True)

        password = st.text_input(
            "🔑 Password",
            type="password",
            placeholder="Enter admin password"
        )

        if st.button(
            "🔐 Login",
            use_container_width=True,
            type="primary"
        ):

            if password.strip() == "admin123":

                st.session_state.attempts = 0

                st.session_state.admin_logged_in = True

                st.success(
                    "✅ Access Granted"
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
                        "🚫 Access blocked after 3 incorrect attempts."
                    )

                    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    CLV Intelligence Dashboard •
    Machine Learning Customer Analytics

    <br>

    Built with Python • Streamlit • Scikit-Learn • Pandas

</div>
""", unsafe_allow_html=True)
