import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEXA CLV | Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "user_inputs.xlsx"


# ============================================================
# STREAMLIT THEME / CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b1020;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Sidebar radio */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 8px 6px;
        border-radius: 8px;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 750;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 650;
    }

    /* Download buttons */
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    /* Inputs */
    div[data-baseweb="input"] {
        border-radius: 9px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }

    /* Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        border-color: #e5e7eb;
        background-color: white;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "last_recency" not in st.session_state:
    st.session_state.last_recency = 30

if "last_frequency" not in st.session_state:
    st.session_state.last_frequency = 5

if "last_monetary" not in st.session_state:
    st.session_state.last_monetary = 500.0

if "admin" not in st.session_state:
    st.session_state.admin = False

if "attempts" not in st.session_state:
    st.session_state.attempts = 0

if "blocked" not in st.session_state:
    st.session_state.blocked = False


# ============================================================
# ML MODEL
# ============================================================

@st.cache_resource
def build_model():

    df = pd.DataFrame({

        "Recency": [
            10, 20, 5, 30,
            15, 40, 25, 8,
            60, 12, 35, 18
        ],

        "Frequency": [
            5, 3, 10, 2,
            7, 1, 4, 12,
            2, 8, 3, 6
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

    X = df[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ]

    y = df["CLV"]

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )

    model.fit(X, y)

    return model


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        return None

    try:
        return pd.read_excel(DATA_FILE)

    except Exception:
        return None


def save_data(row):

    try:

        if os.path.exists(DATA_FILE):

            old = pd.read_excel(DATA_FILE)

            new = pd.concat(
                [old, row],
                ignore_index=True
            )

        else:

            new = row

        new.to_excel(
            DATA_FILE,
            index=False
        )

        return True

    except Exception as e:

        st.error(f"Storage error: {e}")

        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 NEXA CLV")

    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "⌂ Overview",
            "◈ CLV Prediction",
            "◉ Customer Segmentation",
            "▦ Analytics",
            "⚙ Administration"
        ]
    )

    st.divider()

    st.caption("PLATFORM STATUS")

    st.success("● ML ENGINE ONLINE")

    st.caption("Gradient Boosting")

    st.caption("K-Means Clustering")

    st.caption("RFM Analytics")

    st.divider()

    st.caption("NEXA CLV v2.0")


# ============================================================
# OVERVIEW
# ============================================================

if page == "⌂ Overview":

    st.title("NEXA CLV")

    st.subheader(
        "Customer Intelligence powered by Machine Learning"
    )

    st.info(
        "Transform customer behaviour into actionable intelligence "
        "using CLV prediction, RFM analysis and customer segmentation."
    )

    st.divider()

    data = load_data()

    customers = 0
    average = 0
    maximum = 0

    if data is not None and not data.empty:

        customers = len(data)

        if "Predicted_CLV" in data.columns:

            values = pd.to_numeric(
                data["Predicted_CLV"],
                errors="coerce"
            ).dropna()

            if not values.empty:

                average = values.mean()
                maximum = values.max()

    # ========================================================
    # LIVE INTELLIGENCE
    # ========================================================

    st.header("📊 Live Intelligence")

    a, b, c, d = st.columns(4)

    a.metric(
        "Customers Analysed",
        f"{customers:,}"
    )

    b.metric(
        "Average CLV",
        f"${average:,.0f}"
    )

    c.metric(
        "Highest CLV",
        f"${maximum:,.0f}"
    )

    d.metric(
        "ML Engine",
        "ONLINE"
    )

    st.write("")

    # ========================================================
    # FEATURE CARDS
    # ========================================================

    st.header("🚀 Intelligence Modules")

    f1, f2, f3 = st.columns(3)

    with f1:

        with st.container(border=True):

            st.subheader("🧠 ML Prediction")

            st.write(
                "Gradient Boosting predicts the future "
                "Customer Lifetime Value of a customer "
                "from RFM behaviour."
            )

            st.info("Regression Model")

    with f2:

        with st.container(border=True):

            st.subheader("👥 Customer Segmentation")

            st.write(
                "K-Means clustering discovers behavioural "
                "customer groups from Recency, Frequency "
                "and Monetary values."
            )

            st.info("Unsupervised Learning")

    with f3:

        with st.container(border=True):

            st.subheader("📈 Analytics")

            st.write(
                "Analyze predicted customer value using "
                "charts, distributions and historical "
                "prediction records."
            )

            st.info("Business Intelligence")

    st.write("")

    # ========================================================
    # ML PIPELINE
    # ========================================================

    st.header("⚡ How NEXA CLV Works")

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        with st.container(border=True):

            st.subheader("01")

            st.write("📥 Data")

            st.caption(
                "Customer transaction behaviour"
            )

    with p2:

        with st.container(border=True):

            st.subheader("02")

            st.write("📊 RFM")

            st.caption(
                "Recency • Frequency • Monetary"
            )

    with p3:

        with st.container(border=True):

            st.subheader("03")

            st.write("🤖 Model")

            st.caption(
                "Gradient Boosting prediction"
            )

    with p4:

        with st.container(border=True):

            st.subheader("04")

            st.write("💡 Insight")

            st.caption(
                "Customer value intelligence"
            )

    # ========================================================
    # PREVIEW
    # ========================================================

    st.header("📈 Customer Value Preview")

    if (
        data is not None
        and not data.empty
        and "Predicted_CLV" in data.columns
    ):

        values = pd.to_numeric(
            data["Predicted_CLV"],
            errors="coerce"
        ).dropna()

        if not values.empty:

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                range(1, len(values) + 1),
                values,
                marker="o",
                linewidth=2
            )

            ax.set_xlabel(
                "Customer"
            )

            ax.set_ylabel(
                "Predicted CLV"
            )

            ax.set_title(
                "Predicted Customer Lifetime Value"
            )

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

    else:

        st.info(
            "Your CLV intelligence chart will appear "
            "after the first prediction."
        )


# ============================================================
# CLV PREDICTION
# ============================================================

elif page == "◈ CLV Prediction":

    st.title(
        "◈ Customer Lifetime Value Prediction"
    )

    st.write(
        "Enter customer behaviour and let the ML engine "
        "estimate future customer value."
    )

    st.divider()

    left, right = st.columns(
        [1, 1.25]
    )

    # ========================================================
    # INPUT
    # ========================================================

    with left:

        with st.container(border=True):

            st.subheader(
                "👤 Customer Behaviour"
            )

            st.caption(
                "Enter the customer's RFM information."
            )

            recency = st.number_input(
                "Recency",
                min_value=0,
                max_value=365,
                value=st.session_state.last_recency,
                help="Days since last purchase."
            )

            frequency = st.number_input(
                "Frequency",
                min_value=1,
                max_value=100,
                value=st.session_state.last_frequency,
                help="Number of purchases."
            )

            monetary = st.number_input(
                "Monetary Value",
                min_value=0.0,
                max_value=100000.0,
                value=st.session_state.last_monetary,
                step=50.0,
                help="Total amount spent."
            )

            st.write("")

            predict = st.button(
                "🚀 RUN ML PREDICTION",
                type="primary",
                use_container_width=True
            )

    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    with right:

        st.subheader(
            "Customer Profile"
        )

        x1, x2, x3 = st.columns(3)

        x1.metric(
            "RECENCY",
            f"{recency} days"
        )

        x2.metric(
            "FREQUENCY",
            frequency
        )

        x3.metric(
            "MONETARY",
            f"${monetary:,.0f}"
        )

        st.write("")

        with st.container(border=True):

            st.subheader("🤖 Model Information")

            st.write(
                "The NEXA CLV model uses three behavioural "
                "signals:"
            )

            st.write(
                "• Recency — how recently the customer purchased"
            )

            st.write(
                "• Frequency — how often the customer purchases"
            )

            st.write(
                "• Monetary — how much the customer spends"
            )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict:

        model = build_model()

        customer = pd.DataFrame({

            "Recency": [recency],

            "Frequency": [frequency],

            "Monetary": [monetary]

        })

        prediction = float(
            model.predict(customer)[0]
        )

        customer["Predicted_CLV"] = prediction

        save_data(customer)

        st.session_state.prediction = prediction

        st.session_state.last_recency = recency

        st.session_state.last_frequency = frequency

        st.session_state.last_monetary = monetary

    # ========================================================
    # RESULT
    # ========================================================

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction

        st.divider()

        st.header(
            "🎯 ML Prediction Result"
        )

        if prediction >= 2000:

            level = "HIGH VALUE"
            message = (
                "This customer shows strong potential "
                "for long-term business value."
            )

        elif prediction >= 1000:

            level = "MEDIUM VALUE"
            message = (
                "This customer represents a valuable "
                "relationship with growth potential."
            )

        else:

            level = "STANDARD VALUE"
            message = (
                "This customer currently has a lower "
                "predicted lifetime value."
            )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "PREDICTED CLV",
            f"${prediction:,.2f}"
        )

        r2.metric(
            "CUSTOMER SEGMENT",
            level
        )

        r3.metric(
            "MODEL",
            "Gradient Boosting"
        )

        if prediction >= 2000:

            st.success(
                f"🟢 {message}"
            )

        elif prediction >= 1000:

            st.warning(
                f"🟡 {message}"
            )

        else:

            st.info(
                f"🔵 {message}"
            )

        # ====================================================
        # RFM VISUALIZATION
        # ====================================================

        st.header(
            "📊 RFM Behaviour Analysis"
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        features = [
            "Recency",
            "Frequency",
            "Monetary"
        ]

        values = [
            recency,
            frequency,
            monetary
        ]

        ax.bar(
            features,
            values
        )

        ax.set_ylabel(
            "Value"
        )

        ax.set_title(
            "Customer Behaviour Profile"
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


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "◉ Customer Segmentation":

    st.title(
        "◉ Customer Segmentation"
    )

    st.write(
        "Discover behavioural customer groups using "
        "K-Means clustering."
    )

    st.divider()

    st.info(
        "Upload a CSV containing customer Recency, "
        "Frequency and Monetary data."
    )

    uploaded = st.file_uploader(
        "Upload Customer Dataset",
        type=["csv"]
    )

    if uploaded:

        try:

            df = pd.read_csv(
                uploaded
            )

            st.success(
                f"Dataset loaded successfully • "
                f"{len(df):,} customer records"
            )

            with st.expander(
                "👁 Preview Dataset",
                expanded=True
            ):

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

            st.divider()

            st.header(
                "⚙ Feature Selection"
            )

            a, b, c = st.columns(3)

            recency_col = a.selectbox(
                "Recency Column",
                df.columns
            )

            frequency_col = b.selectbox(
                "Frequency Column",
                df.columns
            )

            monetary_col = c.selectbox(
                "Monetary Column",
                df.columns
            )

            cluster_count = st.slider(
                "Number of Customer Clusters",
                min_value=2,
                max_value=6,
                value=3
            )

            run = st.button(
                "🚀 RUN K-MEANS CLUSTERING",
                type="primary",
                use_container_width=True
            )

            if run:

                X = df[
                    [
                        recency_col,
                        frequency_col,
                        monetary_col
                    ]
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                valid = X.notna().all(
                    axis=1
                )

                X = X.loc[valid]

                result = df.loc[
                    valid
                ].copy()

                if len(X) < cluster_count:

                    st.error(
                        "There are not enough valid records "
                        "for the selected number of clusters."
                    )

                else:

                    scaler = StandardScaler()

                    scaled = scaler.fit_transform(
                        X
                    )

                    model = KMeans(
                        n_clusters=cluster_count,
                        random_state=42,
                        n_init=10
                    )

                    result["Cluster"] = model.fit_predict(
                        scaled
                    )

                    st.success(
                        "✓ K-Means segmentation completed successfully."
                    )

                    st.divider()

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric(
                        "Customers",
                        len(result)
                    )

                    m2.metric(
                        "Clusters",
                        cluster_count
                    )

                    m3.metric(
                        "Avg Monetary",
                        f"${X[monetary_col].mean():,.0f}"
                    )

                    m4.metric(
                        "Avg Frequency",
                        f"{X[frequency_col].mean():.1f}"
                    )

                    # =================================================
                    # BEHAVIOURAL MAP
                    # =================================================

                    st.header(
                        "🗺 Behavioural Customer Map"
                    )

                    fig, ax = plt.subplots(
                        figsize=(11, 5)
                    )

                    points = ax.scatter(
                        result[recency_col],
                        result[monetary_col],
                        c=result["Cluster"],
                        cmap="viridis",
                        s=80,
                        alpha=0.85
                    )

                    ax.set_xlabel(
                        recency_col
                    )

                    ax.set_ylabel(
                        monetary_col
                    )

                    ax.set_title(
                        "Customer Behaviour Clusters"
                    )

                    ax.grid(
                        alpha=0.2
                    )

                    plt.colorbar(
                        points,
                        ax=ax,
                        label="Cluster"
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    plt.close(fig)

                    # =================================================
                    # SEGMENT SUMMARY
                    # =================================================

                    st.header(
                        "📋 Segment Intelligence"
                    )

                    summary = result.groupby(
                        "Cluster"
                    )[
                        [
                            recency_col,
                            frequency_col,
                            monetary_col
                        ]
                    ].mean().round(2)

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇ Download Segmented Customers",
                        result.to_csv(
                            index=False
                        ),
                        "customer_segments.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Dataset processing failed: {e}"
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "▦ Analytics":

    st.title(
        "▦ Customer Intelligence Analytics"
    )

    st.write(
        "Monitor the value distribution of predicted customers."
    )

    st.divider()

    data = load_data()

    if data is None or data.empty:

        st.info(
            "No customer predictions are available yet."
        )

    elif "Predicted_CLV" not in data.columns:

        st.warning(
            "Stored data does not contain Predicted_CLV."
        )

    else:

        values = pd.to_numeric(
            data["Predicted_CLV"],
            errors="coerce"
        ).dropna()

        if values.empty:

            st.info(
                "No valid CLV values are available."
            )

        else:

            a, b, c, d = st.columns(4)

            a.metric(
                "Predictions",
                len(values)
            )

            b.metric(
                "Average CLV",
                f"${values.mean():,.0f}"
            )

            c.metric(
                "Maximum CLV",
                f"${values.max():,.0f}"
            )

            d.metric(
                "Minimum CLV",
                f"${values.min():,.0f}"
            )

            st.divider()

            left, right = st.columns(2)

            # =================================================
            # HISTOGRAM
            # =================================================

            with left:

                st.subheader(
                    "📊 CLV Distribution"
                )

                fig, ax = plt.subplots(
                    figsize=(8, 4)
                )

                ax.hist(
                    values,
                    bins=min(10, max(3, len(values)))
                )

                ax.set_xlabel(
                    "Predicted CLV"
                )

                ax.set_ylabel(
                    "Customers"
                )

                ax.set_title(
                    "Customer Lifetime Value Distribution"
                )

                ax.grid(
                    alpha=0.2
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            # =================================================
            # TREND
            # =================================================

            with right:

                st.subheader(
                    "📈 Value Trend"
                )

                fig, ax = plt.subplots(
                    figsize=(8, 4)
                )

                ax.plot(
                    range(1, len(values) + 1),
                    values,
                    marker="o"
                )

                ax.set_xlabel(
                    "Customer"
                )

                ax.set_ylabel(
                    "CLV"
                )

                ax.set_title(
                    "Predicted CLV Trend"
                )

                ax.grid(
                    alpha=0.2
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            st.divider()

            # =================================================
            # VALUE CATEGORIES
            # =================================================

            st.header(
                "🎯 Customer Value Categories"
            )

            high = int(
                (values >= 2000).sum()
            )

            medium = int(
                (
                    (values >= 1000)
                    &
                    (values < 2000)
                ).sum()
            )

            standard = int(
                (values < 1000).sum()
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "🟢 High Value",
                high
            )

            c2.metric(
                "🟡 Medium Value",
                medium
            )

            c3.metric(
                "🔵 Standard Value",
                standard
            )

            st.divider()

            st.header(
                "📋 Prediction Records"
            )

            st.dataframe(
                data,
                use_container_width=True
            )

            st.download_button(
                "⬇ Download Analytics Data",
                data.to_csv(index=False),
                "nexa_clv_analytics.csv",
                "text/csv",
                use_container_width=True
            )


# ============================================================
# ADMINISTRATION
# ============================================================

elif page == "⚙ Administration":

    st.title(
        "⚙ Administration"
    )

    st.write(
        "Secure management of stored prediction records."
    )

    st.divider()

    # ========================================================
    # BLOCKED
    # ========================================================

    if st.session_state.blocked:

        st.error(
            "🔒 Access blocked after 3 incorrect attempts."
        )

        st.write(
            "Please contact the administrator if you "
            "need access."
        )

        st.link_button(
            "📧 Contact Administrator",
            "https://mail.google.com/mail/?view=cm&fs=1&to=atharavshende999@gmail.com",
            use_container_width=True
        )

        st.stop()

    # ========================================================
    # LOGIN
    # ========================================================

    if not st.session_state.admin:

        with st.container(border=True):

            st.subheader(
                "🔐 Administrator Login"
            )

            st.info(
                "Administrator authentication is required "
                "to access stored customer records."
            )

            password = st.text_input(
                "Admin Password",
                type="password"
            )

            login = st.button(
                "🔑 SIGN IN",
                type="primary",
                use_container_width=True
            )

            if login:

                if password == "admin123":

                    st.session_state.admin = True

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
                            f"Incorrect password. "
                            f"{remaining} attempts remaining."
                        )

                    else:

                        st.session_state.blocked = True

                        st.rerun()

    # ========================================================
    # ADMIN DASHBOARD
    # ========================================================

    else:

        st.success(
            "✓ Administrator authenticated successfully."
        )

        if st.button(
            "🚪 LOG OUT",
            use_container_width=True
        ):

            st.session_state.admin = False

            st.rerun()

        data = load_data()

        if data is not None and not data.empty:

            st.divider()

            st.header(
                "📊 Stored Intelligence"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Records",
                len(data)
            )

            if "Predicted_CLV" in data.columns:

                values = pd.to_numeric(
                    data["Predicted_CLV"],
                    errors="coerce"
                ).dropna()

                if not values.empty:

                    b.metric(
                        "Average CLV",
                        f"${values.mean():,.0f}"
                    )

                    c.metric(
                        "Highest CLV",
                        f"${values.max():,.0f}"
                    )

            st.divider()

            st.dataframe(
                data,
                use_container_width=True
            )

            st.download_button(
                "⬇ Download Prediction Records",
                data.to_csv(index=False),
                "clv_records.csv",
                "text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "No prediction records have been stored yet."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NEXA CLV • Machine Learning Customer Intelligence • "
    "Gradient Boosting • K-Means • RFM Analytics"
)
