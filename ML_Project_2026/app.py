# ============================================================
# CUSTOMER LIFETIME VALUE - ML DASHBOARD
# Professional Streamlit UI
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.18);
}

.sidebar-title {
    font-size: 25px;
    font-weight: 800;
}

.sidebar-subtitle {
    font-size: 12px;
    opacity: 0.55;
}

/* MAIN HEADER */

.main-title {
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 4px;
}

.main-subtitle {
    font-size: 16px;
    opacity: 0.60;
    margin-bottom: 25px;
}

/* HERO */

.hero {
    padding: 28px;
    border-radius: 22px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.05);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
}

.hero-text {
    margin-top: 8px;
    font-size: 14px;
    line-height: 1.7;
    opacity: 0.65;
}

/* KPI */

.kpi {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.05);
    min-height: 130px;
}

.kpi-label {
    font-size: 12px;
    font-weight: 700;
    opacity: 0.55;
    letter-spacing: 0.7px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-text {
    font-size: 12px;
    opacity: 0.50;
    margin-top: 5px;
}

/* SECTION */

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 28px;
    margin-bottom: 15px;
}

/* INFO CARDS */

.info-card {
    padding: 22px;
    border-radius: 17px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.04);
    min-height: 160px;
}

.info-title {
    font-size: 18px;
    font-weight: 750;
    margin-bottom: 8px;
}

.info-text {
    font-size: 13px;
    opacity: 0.62;
    line-height: 1.6;
}

/* RESULT */

.prediction-box {
    padding: 28px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,0.20);
    background: rgba(128,128,128,0.05);
    text-align: center;
}

.prediction-label {
    font-size: 13px;
    opacity: 0.55;
}

.prediction-value {
    font-size: 42px;
    font-weight: 850;
    margin-top: 5px;
}

/* BUTTON */

.stButton > button {
    border-radius: 11px;
    min-height: 45px;
    font-weight: 700;
}

/* FOOTER */

.footer {
    text-align: center;
    font-size: 12px;
    opacity: 0.45;
    padding: 15px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BUILT-IN DATASET
# ============================================================

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


# ============================================================
# TRAIN ML MODEL
# ============================================================

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
    test_size=0.20,
    random_state=42
)


model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            📊 CLV Intelligence
        </div>

        <div class="sidebar-subtitle">
            Machine Learning Customer Analytics
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    st.success("🟢 ML MODEL ONLINE")

    st.markdown("### Navigation")

    menu = st.radio(
        "",
        [
            "🏠 Home",
            "🔮 CLV Predictor",
            "📊 Segmentation Dashboard",
            "🔐 Admin Panel"
        ]
    )

    st.divider()

    st.caption("MACHINE LEARNING")

    st.write("Gradient Boosting")

    st.caption("SEGMENTATION")

    st.write("K-Means Clustering")

    st.caption("DATA")

    st.write("Built-in Customer Dataset")

    st.divider()

    st.caption(
        "Python • Pandas • Scikit-learn • Streamlit"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Customer Lifetime Value
    </div>

    <div class="main-subtitle">
        ML-powered customer intelligence & prediction platform
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HOME
# ============================================================

if menu == "🏠 Home":

    # HERO IMAGE
    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1600&q=80",
        use_container_width=True
    )

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
            🚀 Intelligent Customer Analytics
        </div>

        <div class="hero-text">
            Customer Lifetime Value prediction helps businesses
            understand which customers are likely to generate
            greater value in the future. This application combines
            Machine Learning prediction with customer segmentation
            to provide actionable business insights.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # KPI
    # ========================================================

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="kpi">

            <div class="kpi-label">
            CUSTOMERS
            </div>

            <div class="kpi-value">
            {len(df)}
            </div>

            <div class="kpi-text">
            Customer records analyzed
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="kpi">

            <div class="kpi-label">
            AVERAGE CLV
            </div>

            <div class="kpi-value">
            ${df["CLV"].mean():,.0f}
            </div>

            <div class="kpi-text">
            Average customer value
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="kpi">

            <div class="kpi-label">
            HIGHEST CLV
            </div>

            <div class="kpi-value">
            ${df["CLV"].max():,.0f}
            </div>

            <div class="kpi-text">
            Maximum customer value
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
            <div class="kpi">

            <div class="kpi-label">
            ML ALGORITHM
            </div>

            <div class="kpi-value">
            GB
            </div>

            <div class="kpi-text">
            Gradient Boosting Regressor
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # CLV CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">Customer Value Analysis</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2)

    with chart1:

        st.markdown("### 📈 CLV Distribution")

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.hist(
            df["CLV"],
            bins=7,
            alpha=0.75
        )

        ax.set_xlabel("Customer Lifetime Value")

        ax.set_ylabel("Customers")

        ax.set_title(
            "Distribution of Customer Value"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with chart2:

        st.markdown("### 💰 Customer Value")

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            range(1, len(df) + 1),
            df["CLV"]
        )

        ax.set_xlabel("Customer")

        ax.set_ylabel("CLV")

        ax.set_title(
            "Customer Lifetime Value"
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown(
        '<div class="section-title">Platform Capabilities</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-title">
            🔮 CLV Prediction
            </div>

            <div class="info-text">
            Estimate customer lifetime value using
            Recency, Frequency and Monetary behavior
            with a Gradient Boosting model.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-title">
            🎯 Customer Segmentation
            </div>

            <div class="info-text">
            K-Means clustering automatically discovers
            groups of customers with similar behavior.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-title">
            🔐 Admin Management
            </div>

            <div class="info-text">
            Protected administration section for
            accessing customer information.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # DATA
    # ========================================================

    st.markdown(
        '<div class="section-title">Customer Dataset</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CLV PREDICTOR
# ============================================================

elif menu == "🔮 CLV Predictor":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🔮 Predict Customer Lifetime Value
        </div>

        <div class="hero-text">
        Enter customer purchase behavior and let the
        Machine Learning model estimate future customer value.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        recency = st.number_input(
            "🕒 Recency (Days)",
            min_value=0,
            max_value=365,
            value=30
        )

    with c2:

        frequency = st.number_input(
            "🔄 Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5
        )

    with c3:

        monetary = st.number_input(
            "💰 Monetary Value",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=50.0
        )

    st.write("")

    if st.button(
        "🚀 Predict Customer Lifetime Value",
        use_container_width=True
    ):

        user_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = model.predict(
            user_data
        )[0]

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="prediction-box">

            <div class="prediction-label">
            PREDICTED CUSTOMER LIFETIME VALUE
            </div>

            <div class="prediction-value">
            ${prediction:,.2f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if prediction >= df["CLV"].quantile(0.75):

            st.success(
                "⭐ HIGH-VALUE CUSTOMER\n\n"
                "Recommended for premium retention and loyalty campaigns."
            )

        elif prediction >= df["CLV"].median():

            st.info(
                "📈 MEDIUM-VALUE CUSTOMER\n\n"
                "Suitable for engagement and personalized campaigns."
            )

        else:

            st.warning(
                "📌 LOWER-VALUE CUSTOMER\n\n"
                "Consider targeted marketing strategies."
            )

        st.markdown("### Customer Input")

        result = pd.DataFrame({
            "Parameter": [
                "Recency",
                "Frequency",
                "Monetary",
                "Predicted CLV"
            ],

            "Value": [
                f"{recency} days",
                frequency,
                f"${monetary:,.2f}",
                f"${prediction:,.2f}"
            ]
        })

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SEGMENTATION
# ============================================================

elif menu == "📊 Segmentation Dashboard":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🎯 Customer Segmentation
        </div>

        <div class="hero-text">
        Discover customer groups using K-Means clustering.
        The application automatically uses the built-in customer
        dataset, so no file upload is required.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    clusters = st.slider(
        "Number of Customer Segments",
        2,
        5,
        3
    )

    if st.button(
        "🚀 Run Segmentation",
        use_container_width=True
    ):

        segment_features = df[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            segment_features
        )

        kmeans = KMeans(
            n_clusters=clusters,
            random_state=42,
            n_init=10
        )

        segmented = df.copy()

        segmented["Cluster"] = (
            kmeans.fit_predict(scaled) + 1
        )

        st.success(
            f"Successfully created {clusters} customer segments."
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Customer Segment Map"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.scatter(
            segmented["Recency"],
            segmented["Monetary"],
            c=segmented["Cluster"],
            s=130,
            alpha=0.8
        )

        ax.set_xlabel(
            "Recency"
        )

        ax.set_ylabel(
            "Monetary Value"
        )

        ax.set_title(
            "Customer Segmentation using K-Means"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.markdown(
            "### 📋 Segment Summary"
        )

        summary = (
            segmented
            .groupby("Cluster")
            .agg(
                Customers=("CLV", "count"),
                Average_CLV=("CLV", "mean"),
                Average_Frequency=("Frequency", "mean"),
                Average_Monetary=("Monetary", "mean")
            )
            .reset_index()
        )

        st.dataframe(
            summary.style.format({
                "Average_CLV": "${:,.2f}",
                "Average_Frequency": "{:.2f}",
                "Average_Monetary": "${:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "### 👥 Segmented Customers"
        )

        st.dataframe(
            segmented,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ADMIN
# ============================================================

elif menu == "🔐 Admin Panel":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🔐 Administrator Access
        </div>

        <div class="hero-text">
        Restricted area for authorized project administrators.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if "attempts" not in st.session_state:

        st.session_state.attempts = 0

    if "blocked" not in st.session_state:

        st.session_state.blocked = False

    # --------------------------------------------------------
    # BLOCKED
    # --------------------------------------------------------

    if st.session_state.blocked:

        st.error(
            "🚫 Access blocked after 3 incorrect attempts."
        )

        email = "atharavshende999@gmail.com"

        subject = "Access Request for CLV App"

        gmail_link = (
            "https://mail.google.com/mail/"
            "?view=cm&fs=1"
            f"&to={email}"
            f"&su={subject}"
        )

        st.markdown(
            f<a href="{gmail_link}" target="_blank">

            <div style="
                display:inline-block;
                padding:13px 25px;
                background:#ff4b4b;
                color:white;
                font-weight:bold;
                border-radius:10px;
                text-decoration:none;
        >

            📧 Contact Admin

            </div>

            </a>
            ,
            unsafe_allow_html=True
        )

        st.stop()

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    st.markdown(
        "### 🔑 Secure Login"
    )

    password = st.text_input(
        "Enter Administrator Password",
        type="password"
    )

    if st.button(
        "🔓 Login",
        use_container_width=True
    ):

        if password.strip() == "admin123":

            st.session_state.attempts = 0

            st.success(
                "✅ Access Granted"
            )

            st.markdown(
                "### 📊 Admin Dashboard"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Customers",
                    len(df)
                )

            with c2:

                st.metric(
                    "Average CLV",
                    f"${df['CLV'].mean():,.0f}"
                )

            with c3:

                st.metric(
                    "Maximum CLV",
                    f"${df['CLV'].max():,.0f}"
                )

            st.divider()

            st.markdown(
                "### 📄 Customer Records"
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            csv = df.to_csv(
                index=False
            )

            st.download_button(
                "📥 Download Customer Data",
                csv,
                "customer_clv_data.csv",
                "text/csv",
                use_container_width=True
            )

        else:

            st.session_state.attempts += 1

            remaining = (
                3 -
                st.session_state.attempts
            )

            if remaining > 0:

                st.error(
                    f"❌ Wrong password! "
                    f"Attempts remaining: {remaining}"
                )

            else:

                st.session_state.blocked = True

                st.error(
                    "🚫 You are blocked after 3 wrong attempts."
                )

                st.rerun()



st.divider()

st.markdown(
    <div class="footer">

    📊 Customer Lifetime Value Prediction
    &nbsp; • &nbsp;
    Machine Learning Project
    &nbsp; • &nbsp;
    Python + Scikit-learn + Streamlit

    </div>
    unsafe_allow_html=True
)
