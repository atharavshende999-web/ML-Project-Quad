import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CLV Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Cards */
    .dashboard-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }

    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Prediction result */
    .prediction-card {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(79,70,229,0.25);
        margin-top: 25px;
    }

    .prediction-title {
        font-size: 17px;
        opacity: 0.9;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 800;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: none;
        padding: 10px 20px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 10px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center; padding:15px 0 25px 0;">
        <div style="font-size:48px;">📊</div>
        <h2 style="margin:0;">CLV Analytics</h2>
        <p style="color:#cbd5e1 !important;">
            Customer Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🔮 CLV Predictor",
            "📊 Segmentation Dashboard",
            "🔐 Admin Panel"
        ]
    )

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center; color:#cbd5e1;">
        <small>AI Powered Customer Analytics</small>
        <br>
        <small>Python • ML • Streamlit</small>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HOME
# =========================================================

if menu == "🏠 Home":

    st.markdown(
        '<div class="main-title">Customer Lifetime Value</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered customer analytics and segmentation platform'
        '</div>',
        unsafe_allow_html=True
    )

    # Hero
    st.markdown("""
    <div class="dashboard-card">

        <h2>🚀 Welcome to CLV Analytics</h2>

        <p style="font-size:16px;color:#64748b;">
        This application uses Machine Learning to estimate Customer
        Lifetime Value and identify meaningful customer segments.
        </p>

    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🤖 ML MODEL</div>
            <div class="metric-value">GBR</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">👥 SEGMENTS</div>
            <div class="metric-value">3</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">📈 FEATURES</div>
            <div class="metric-value">3</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🔐 SECURITY</div>
            <div class="metric-value">ADMIN</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    st.markdown(
        '<div class="section-title">✨ Platform Features</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="dashboard-card">

        <h3>🔮 CLV Prediction</h3>

        <p style="color:#64748b;">
        Predict the future value of a customer using
        Recency, Frequency and Monetary behaviour.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="dashboard-card">

        <h3>🔐 Secure Administration</h3>

        <p style="color:#64748b;">
        Protected admin panel with login attempts,
        stored data access and Excel download.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dashboard-card">

        <h3>📊 Customer Segmentation</h3>

        <p style="color:#64748b;">
        Automatically group customers using K-Means
        clustering based on their behaviour.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="dashboard-card">

        <h3>💾 Data Storage</h3>

        <p style="color:#64748b;">
        Prediction inputs are securely stored in an
        Excel file for administrative analysis.
        </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# CLV PREDICTOR
# =========================================================

elif menu == "🔮 CLV Predictor":

    st.markdown(
        '<div class="main-title">🔮 CLV Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Estimate the Customer Lifetime Value using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Customer Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "📅 Recency (Days)",
            min_value=0,
            max_value=365,
            value=30
        )

    with col2:
        frequency = st.number_input(
            "🛒 Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5
        )

    with col3:
        monetary = st.number_input(
            "💰 Monetary Value",
            min_value=0.0,
            max_value=10000.0,
            value=500.0
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button(
        "🚀 Predict Customer Lifetime Value",
        use_container_width=True
    ):

        data = {
            "Recency": [10,20,5,30,15,40,25,8,60,12,35,18],
            "Frequency": [5,3,10,2,7,1,4,12,2,8,3,6],
            "Monetary": [500,300,1000,200,700,100,400,1500,250,900,350,650],
            "CLV": [1200,700,2500,400,1600,200,900,3000,500,2000,800,1400]
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

        user_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = model.predict(user_data)

        predicted_value = prediction[0]

        user_data["Predicted_CLV"] = predicted_value

        file_name = "user_inputs.xlsx"

        try:
            old = pd.read_excel(file_name)

            new = pd.concat(
                [old, user_data],
                ignore_index=True
            )

            new.to_excel(
                file_name,
                index=False
            )

        except:
            user_data.to_excel(
                file_name,
                index=False
            )

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-title">
                    💰 Estimated Customer Lifetime Value
                </div>

                <div class="prediction-value">
                    ${predicted_value:,.2f}
                </div>

                <div style="margin-top:10px;">
                    Prediction generated using Gradient Boosting ML
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "✅ Customer prediction successfully generated and stored."
        )


# =========================================================
# SEGMENTATION
# =========================================================

elif menu == "📊 Segmentation Dashboard":

    st.markdown(
        '<div class="main-title">📊 Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover customer groups using K-Means clustering'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-card">',
        unsafe_allow_html=True
    )

    file = st.file_uploader(
        "📁 Upload Customer CSV Dataset",
        type=["csv"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if file:

        df = pd.read_csv(file)

        st.markdown(
            '<div class="section-title">📋 Dataset Preview</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">⚙️ Select Features</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            r = st.selectbox(
                "📅 Recency Column",
                df.columns
            )

        with col2:
            f = st.selectbox(
                "🛒 Frequency Column",
                df.columns
            )

        with col3:
            m = st.selectbox(
                "💰 Monetary Column",
                df.columns
            )

        if st.button(
            "🚀 Run Customer Segmentation",
            use_container_width=True
        ):

            X = df[[r, f, m]]

            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            )

            df["Cluster"] = kmeans.fit_predict(
                X_scaled
            )

            st.success(
                "✅ Customer segmentation completed successfully."
            )

            # Metrics
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "👥 Total Customers",
                    len(df)
                )

            with c2:
                st.metric(
                    "🎯 Number of Segments",
                    df["Cluster"].nunique()
                )

            with c3:
                st.metric(
                    "📊 Features Used",
                    3
                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title">'
                '👥 Segmented Customers'
                '</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                df.head(20),
                use_container_width=True
            )

            # Chart
            st.markdown(
                '<div class="section-title">'
                '📈 Customer Segment Visualization'
                '</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            scatter = ax.scatter(
                df[r],
                df[m],
                c=df["Cluster"],
                s=80,
                alpha=0.8
            )

            ax.set_xlabel(
                "Recency",
                fontsize=12
            )

            ax.set_ylabel(
                "Monetary Value",
                fontsize=12
            )

            ax.set_title(
                "Customer Segmentation",
                fontsize=16,
                fontweight="bold"
            )

            ax.grid(
                alpha=0.2
            )

            st.pyplot(fig)


# =========================================================
# ADMIN PANEL
# =========================================================

elif menu == "🔐 Admin Panel":

    st.markdown(
        '<div class="main-title">🔐 Admin Panel</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Secure access to customer prediction data'
        '</div>',
        unsafe_allow_html=True
    )

    if "attempts" not in st.session_state:
        st.session_state.attempts = 0

    if "blocked" not in st.session_state:
        st.session_state.blocked = False

    # BLOCKED
    if st.session_state.blocked:

        st.error(
            "🚫 Too many incorrect attempts. Access has been blocked."
        )

        st.markdown(
            """
            <div class="dashboard-card">

            <h3>📩 Contact Administrator</h3>

            <p style="color:#64748b;">
            Contact the administrator to request access
            to the CLV dashboard.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        email = "atharavshende999@gmail.com"

        subject = "Access Request for CLV App"

        gmail_link = (
            f"https://mail.google.com/mail/?view=cm"
            f"&fs=1&to={email}&su={subject}"
        )

        st.markdown(
            f"""
            <a href="{gmail_link}" target="_blank">
                <div style="
                    display:inline-block;
                    padding:13px 28px;
                    background:linear-gradient(
                        135deg,#ef4444,#dc2626
                    );
                    color:white;
                    font-weight:bold;
                    border-radius:10px;
                    text-align:center;
                    text-decoration:none;
                ">
                    📧 Contact Admin
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )

        st.stop()

    # LOGIN CARD
    st.markdown(
        '<div class="dashboard-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        '🛡️ Administrator Login'
        '</div>',
        unsafe_allow_html=True
    )

    password = st.text_input(
        "🔑 Enter Admin Password",
        type="password",
        placeholder="Enter your password"
    )

    login = st.button(
        "🔓 Login",
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    if login:

        if password.strip() == "admin123":

            st.session_state.attempts = 0

            st.success(
                "✅ Access Granted"
            )

            st.markdown(
                '<div class="section-title">'
                '📂 Server Files'
                '</div>',
                unsafe_allow_html=True
            )

            files = os.listdir()

            st.dataframe(
                pd.DataFrame(
                    {"Files": files}
                ),
                use_container_width=True
            )

            try:

                df = pd.read_excel(
                    "user_inputs.xlsx"
                )

                st.markdown(
                    '<div class="section-title">'
                    '📄 Stored Prediction Data'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.metric(
                    "👥 Total Predictions",
                    len(df)
                )

            except:

                st.warning(
                    "⚠️ No prediction data found."
                )

            try:

                with open(
                    "user_inputs.xlsx",
                    "rb"
                ) as f:

                    st.download_button(
                        "📥 Download Excel Data",
                        f,
                        file_name="user_inputs.xlsx",
                        use_container_width=True
                    )

            except:

                pass

        else:

            st.session_state.attempts += 1

            remaining = (
                3 - st.session_state.attempts
            )

            if remaining > 0:

                st.error(
                    f"❌ Incorrect password! "
                    f"Attempts remaining: {remaining}"
                )

            else:

                st.session_state.blocked = True

                st.error(
                    "🚫 Access blocked after 3 incorrect attempts."
                )

                st.rerun()
