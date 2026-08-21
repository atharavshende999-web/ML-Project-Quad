import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="CLV Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "user_inputs.xlsx"


# =========================================================
# THEME / NATIVE STREAMLIT CSS
# =========================================================
# This CSS is only for visual styling of Streamlit components.
# No raw HTML is rendered into the application content.

st.markdown("""
<style>
    .stApp {
        background: #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .hero-box {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e3a8a 55%,
            #4f46e5 100%
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero-box h1 {
        color: white;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero-box p {
        color: #dbeafe;
        font-size: 17px;
    }

    .small-label {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #111827;
    }

    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        min-height: 180px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .feature-icon {
        font-size: 30px;
    }

    .feature-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 10px;
    }

    .feature-text {
        color: #64748b;
        font-size: 14px;
    }

    .result-box {
        background: white;
        border: 2px solid #c7d2fe;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 5px 20px rgba(79, 70, 229, 0.08);
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 25px;
        font-size: 13px;
    }

    button[kind="primary"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "admin_logged_in": False,
    "attempts": 0,
    "blocked": False,
    "last_prediction": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def train_model():

    training_data = pd.DataFrame({
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
    })

    X = training_data[
        ["Recency", "Frequency", "Monetary"]
    ]

    y = training_data["CLV"]

    model = GradientBoostingRegressor(
        random_state=42
    )

    model.fit(X, y)

    return model


# =========================================================
# DATA FUNCTIONS
# =========================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        return None

    try:
        return pd.read_excel(DATA_FILE)
    except:
        return None


def save_prediction(record):

    try:

        if os.path.exists(DATA_FILE):

            old = pd.read_excel(DATA_FILE)

            final = pd.concat(
                [old, record],
                ignore_index=True
            )

        else:
            final = record

        final.to_excel(
            DATA_FILE,
            index=False
        )

        return True

    except Exception as e:

        st.error(f"Could not save data: {e}")

        return False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ✦ CLV")
    st.caption("INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "CLV Predictor",
            "Customer Segments",
            "Analytics",
            "Admin"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.success("● SYSTEM ONLINE")

    st.caption("AI Engine: Active")
    st.caption("Analytics: Ready")
    st.caption("Data Storage: Ready")

    st.divider()

    st.caption("CLV Intelligence v1.0")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="hero-box">'
        '<div class="small-label">AI CUSTOMER INTELLIGENCE</div>'
        '<h1>Understand the value<br>behind every customer.</h1>'
        '<p>Predict Customer Lifetime Value, discover customer segments, '
        'and turn purchasing behaviour into actionable insights.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    data = load_data()

    customers = 0
    avg_clv = 0
    max_clv = 0

    if data is not None and not data.empty:

        customers = len(data)

        if "Predicted_CLV" in data.columns:

            avg_clv = data["Predicted_CLV"].mean()
            max_clv = data["Predicted_CLV"].max()

    st.markdown(
        '<div class="section-title">Business Snapshot</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "👥 Customers Analysed",
        customers
    )

    b.metric(
        "💰 Average CLV",
        f"${avg_clv:,.0f}"
    )

    c.metric(
        "🏆 Highest CLV",
        f"${max_clv:,.0f}"
    )

    d.metric(
        "⚡ ML Status",
        "Active"
    )

    st.write("")

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">What you can do</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <div class="feature-title">Predict CLV</div>
            <div class="feature-text">
                Estimate the future value of a customer
                from Recency, Frequency and Monetary behaviour.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">👥</div>
            <div class="feature-title">Segment Customers</div>
            <div class="feature-text">
                Use K-Means clustering to identify meaningful
                customer groups.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Explore Analytics</div>
            <div class="feature-text">
                Visualise customer value and monitor prediction
                trends through interactive analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # -----------------------------------------------------
    # ANALYTICS PREVIEW
    # -----------------------------------------------------

    left, right = st.columns([1.4, 1])

    with left:

        st.markdown(
            '<div class="section-title">CLV Performance</div>',
            unsafe_allow_html=True
        )

        if data is not None and "Predicted_CLV" in data.columns:

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.plot(
                data["Predicted_CLV"],
                marker="o"
            )

            ax.set_title(
                "Customer Lifetime Value Trend"
            )

            ax.set_xlabel("Customer")
            ax.set_ylabel("Predicted CLV")

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Generate your first CLV prediction to see analytics here."
            )

    with right:

        st.markdown(
            '<div class="section-title">RFM Intelligence</div>',
            unsafe_allow_html=True
        )

        st.info(
            """
            **Recency**

            Measures how recently a customer purchased.

            **Frequency**

            Measures how often a customer purchases.

            **Monetary**

            Measures how much the customer spends.

            Together, RFM behaviour helps estimate customer value.
            """
        )

    st.divider()

    st.markdown(
        '<div class="section-title">Machine Learning Pipeline</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.info("### 01\nCustomer Data")
    p2.info("### 02\nRFM Analysis")
    p3.info("### 03\nML Prediction")
    p4.info("### 04\nBusiness Insight")


# =========================================================
# CLV PREDICTOR
# =========================================================

elif page == "CLV Predictor":

    st.markdown(
        '<div class="main-title">CLV Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Estimate customer lifetime value using RFM behaviour.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 1.15]
    )

    with left:

        st.markdown(
            '<div class="section-title">Customer Profile</div>',
            unsafe_allow_html=True
        )

        recency = st.number_input(
            "📅 Recency — Days since last purchase",
            min_value=0,
            max_value=365,
            value=30
        )

        frequency = st.number_input(
            "🔄 Frequency — Number of purchases",
            min_value=1,
            max_value=100,
            value=5
        )

        monetary = st.number_input(
            "💰 Monetary — Total spending",
            min_value=0.0,
            max_value=100000.0,
            value=500.0,
            step=50.0
        )

        st.write("")

        predict = st.button(
            "🚀 Calculate Customer Value",
            type="primary",
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section-title">RFM Profile</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Recency",
            f"{recency} d"
        )

        m2.metric(
            "Frequency",
            frequency
        )

        m3.metric(
            "Monetary",
            f"${monetary:,.0f}"
        )

        st.info(
            "The model uses these three customer behaviour "
            "signals to estimate future customer value."
        )

        st.progress(
            min(recency / 365, 1.0)
        )

        st.caption(
            "Recency scale"
        )

    if predict:

        model = train_model()

        user_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = float(
            model.predict(user_data)[0]
        )

        user_data["Predicted_CLV"] = prediction

        save_prediction(user_data)

        st.session_state.last_prediction = prediction

    if st.session_state.last_prediction is not None:

        prediction = st.session_state.last_prediction

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction >= 2000:
            category = "🟢 HIGH VALUE"
        elif prediction >= 1000:
            category = "🟡 MEDIUM VALUE"
        else:
            category = "🔵 STANDARD VALUE"

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Estimated CLV",
            f"${prediction:,.2f}"
        )

        r2.metric(
            "Customer Segment",
            category
        )

        r3.metric(
            "Prediction Engine",
            "Gradient Boosting"
        )

        st.success(
            "Prediction completed successfully and stored."
        )

        st.markdown(
            '<div class="section-title">Customer Behaviour Profile</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.bar(
            ["Recency", "Frequency", "Monetary"],
            [recency, frequency, monetary]
        )

        ax.set_ylabel("Value")
        ax.set_title("RFM Customer Profile")
        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


# =========================================================
# CUSTOMER SEGMENTS
# =========================================================

elif page == "Customer Segments":

    st.markdown(
        '<div class="main-title">Customer Segments</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover hidden customer groups using K-Means clustering.'
        '</div>',
        unsafe_allow_html=True
    )

    file = st.file_uploader(
        "📂 Upload customer CSV",
        type=["csv"]
    )

    if file:

        try:

            df = pd.read_csv(file)

            st.success(
                f"Dataset loaded successfully • {len(df)} records"
            )

            st.dataframe(
                df.head(8),
                use_container_width=True
            )

            st.divider()

            st.markdown(
                '<div class="section-title">Configure RFM Segmentation</div>',
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)

            r = a.selectbox(
                "Recency column",
                df.columns
            )

            f = b.selectbox(
                "Frequency column",
                df.columns
            )

            m = c.selectbox(
                "Monetary column",
                df.columns
            )

            clusters = st.slider(
                "Number of customer segments",
                2,
                6,
                3
            )

            run = st.button(
                "🚀 Analyse Customer Segments",
                type="primary",
                use_container_width=True
            )

            if run:

                X = df[
                    [r, f, m]
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                valid = X.notna().all(
                    axis=1
                )

                X = X[valid]

                result = df.loc[
                    valid
                ].copy()

                if len(X) < clusters:

                    st.error(
                        "Not enough valid customer records."
                    )

                else:

                    scaler = StandardScaler()

                    scaled = scaler.fit_transform(
                        X
                    )

                    kmeans = KMeans(
                        n_clusters=clusters,
                        random_state=42,
                        n_init=10
                    )

                    result["Cluster"] = (
                        kmeans.fit_predict(scaled)
                    )

                    st.success(
                        "Customer segmentation completed."
                    )

                    x1, x2, x3, x4 = st.columns(4)

                    x1.metric(
                        "Customers",
                        len(result)
                    )

                    x2.metric(
                        "Segments",
                        clusters
                    )

                    x3.metric(
                        "Avg Spending",
                        f"${X[m].mean():,.0f}"
                    )

                    x4.metric(
                        "Avg Recency",
                        f"{X[r].mean():.1f}"
                    )

                    st.divider()

                    st.markdown(
                        '<div class="section-title">Customer Map</div>',
                        unsafe_allow_html=True
                    )

                    fig, ax = plt.subplots(
                        figsize=(11, 5)
                    )

                    scatter = ax.scatter(
                        result[r],
                        result[m],
                        c=result["Cluster"],
                        cmap="viridis",
                        s=90,
                        alpha=0.8
                    )

                    ax.set_xlabel(r)
                    ax.set_ylabel(m)
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

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    st.markdown(
                        '<div class="section-title">Segment Summary</div>',
                        unsafe_allow_html=True
                    )

                    summary = result.groupby(
                        "Cluster"
                    )[
                        [r, f, m]
                    ].mean().round(2)

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )

                    st.markdown(
                        '<div class="section-title">Classified Customers</div>',
                        unsafe_allow_html=True
                    )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Download Segmented CSV",
                        result.to_csv(
                            index=False
                        ),
                        "segmented_customers.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Could not process dataset: {e}"
            )

    else:

        st.info(
            "Upload your customer CSV above to begin segmentation."
        )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.markdown(
        '<div class="main-title">Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor customer value and prediction performance.'
        '</div>',
        unsafe_allow_html=True
    )

    data = load_data()

    if data is None or data.empty:

        st.info(
            "No prediction data available. "
            "Make a prediction first."
        )

    elif "Predicted_CLV" not in data.columns:

        st.warning(
            "No CLV prediction column found."
        )

    else:

        values = data["Predicted_CLV"]

        a, b, c, d = st.columns(4)

        a.metric(
            "👥 Predictions",
            len(values)
        )

        b.metric(
            "💰 Average",
            f"${values.mean():,.0f}"
        )

        c.metric(
            "🏆 Highest",
            f"${values.max():,.0f}"
        )

        d.metric(
            "📉 Lowest",
            f"${values.min():,.0f}"
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.markdown(
                '<div class="section-title">CLV Distribution</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.hist(
                values,
                bins=10
            )

            ax.set_xlabel(
                "Predicted CLV"
            )

            ax.set_ylabel(
                "Customers"
            )

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        with right:

            st.markdown(
                '<div class="section-title">CLV Trend</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.plot(
                values,
                marker="o"
            )

            ax.set_xlabel(
                "Prediction"
            )

            ax.set_ylabel(
                "CLV"
            )

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Records</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            data,
            use_container_width=True
        )


# =========================================================
# ADMIN
# =========================================================

elif page == "Admin":

    st.markdown(
        '<div class="main-title">Admin Control Center</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Manage stored customer prediction data.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    if st.session_state.blocked:

        st.error(
            "🚫 Access blocked after 3 incorrect attempts."
        )

        st.link_button(
            "📧 Contact Administrator",
            "https://mail.google.com/mail/?view=cm&fs=1&to=atharavshende999@gmail.com",
            use_container_width=True
        )

        st.stop()

    if not st.session_state.admin_logged_in:

        st.info(
            "🔐 Administrator authentication required."
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login = st.button(
            "🔓 Sign In",
            type="primary",
            use_container_width=True
        )

        if login:

            if password == "admin123":

                st.session_state.admin_logged_in = True
                st.session_state.attempts = 0

                st.rerun()

            else:

                st.session_state.attempts += 1

                remaining = (
                    3 -
                    st.session_state.attempts
                )

                if remaining > 0:

                    st.error(
                        f"Incorrect password • "
                        f"{remaining} attempts remaining"
                    )

                else:

                    st.session_state.blocked = True

                    st.rerun()

    else:

        st.success(
            "🟢 Administrator authenticated"
        )

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.admin_logged_in = False

            st.rerun()

        data = load_data()

        if data is not None and not data.empty:

            st.divider()

            st.markdown(
                '<div class="section-title">Stored Customer Data</div>',
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)

            a.metric(
                "Records",
                len(data)
            )

            if "Predicted_CLV" in data.columns:

                b.metric(
                    "Average CLV",
                    f"${data['Predicted_CLV'].mean():,.0f}"
                )

                c.metric(
                    "Highest CLV",
                    f"${data['Predicted_CLV'].max():,.0f}"
                )

            st.dataframe(
                data,
                use_container_width=True
            )

            st.download_button(
                "📥 Download Customer Excel",
                data.to_csv(index=False),
                "customer_predictions.csv",
                "text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "No customer data has been stored yet."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '✦ CLV Intelligence • Customer Lifetime Value Analytics<br>'
    'Built with Python • Streamlit • Scikit-Learn'
    '</div>',
    unsafe_allow_html=True
)
