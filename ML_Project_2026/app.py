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
# CURRENCY CONVERSION
# ============================================================

# 1 USD = 95.70 INR
USD_TO_INR = 95.70


def usd_to_inr(value):
    """Convert USD value to Indian Rupees."""
    return float(value) * USD_TO_INR


def format_inr(value):
    """Format value as Indian Rupees."""
    return f"₹{float(value):,.2f}"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
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

    /* PREDICTION */

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

    .prediction-usd {
        font-size: 15px;
        opacity: 0.55;
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# ORIGINAL BUILT-IN DATASET
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

    # Original model target is in USD
    "CLV": [
        1200, 700, 2500, 400,
        1600, 200, 900, 3000,
        500, 2000, 800, 1400
    ]
}


base_df = pd.DataFrame(data)


# ============================================================
# ADD CUSTOMER IDs
# ============================================================

base_df.insert(
    0,
    "Customer_ID",
    [f"CUST-{i:04d}" for i in range(1, len(base_df) + 1)]
)

base_df["Source"] = "Original Dataset"


# ============================================================
# SESSION STATE
# ============================================================

# IMPORTANT:
# This stores all customers, including newly predicted customers.
# It prevents the dashboard from going back to only 12 customers
# after Streamlit reruns.

if "customers" not in st.session_state:

    st.session_state.customers = base_df.copy()


if "last_prediction" not in st.session_state:

    st.session_state.last_prediction = None


if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


if "admin_logged_in" not in st.session_state:

    st.session_state.admin_logged_in = False


if "attempts" not in st.session_state:

    st.session_state.attempts = 0


if "blocked" not in st.session_state:

    st.session_state.blocked = False


# ============================================================
# CURRENT CUSTOMER DATA
# ============================================================

df = st.session_state.customers.copy()


# ============================================================
# TRAIN ML MODEL
# ============================================================

X = base_df[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

y = base_df["CLV"]


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


model.fit(
    X_train,
    y_train
)


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

    st.caption("CUSTOMER RECORDS")

    st.write(f"👥 **{len(df)} Customers**")

    st.caption("NEW PREDICTIONS")

    predicted_count = len(
        df[df["Source"] == "Predicted"]
    )

    st.write(f"🔮 **{predicted_count} Predicted**")

    st.caption("MACHINE LEARNING")

    st.write("Gradient Boosting")

    st.caption("SEGMENTATION")

    st.write("K-Means Clustering")

    st.caption("CURRENCY")

    st.write("🇮🇳 Indian Rupee (INR)")

    st.caption("EXCHANGE RATE")

    st.write(f"1 USD = ₹{USD_TO_INR}")

    st.divider()

    # Reset button
    if st.button(
        "🔄 Reset Customer Records",
        use_container_width=True
    ):

        st.session_state.customers = base_df.copy()
        st.session_state.last_prediction = None
        st.session_state.prediction_history = []

        st.success("Customer records reset to 12.")

        st.rerun()

    st.divider()

    st.caption(
        "Python • Pandas • Scikit-learn • Streamlit"
    )


# ============================================================
# MAIN HEADER
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


    # CUSTOMER COUNT
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


    # AVERAGE CLV
    with c2:

        avg_clv_inr = usd_to_inr(
            df["CLV"].mean()
        )

        st.markdown(
            f"""
            <div class="kpi">

            <div class="kpi-label">
            AVERAGE CLV
            </div>

            <div class="kpi-value">
            ₹{avg_clv_inr:,.0f}
            </div>

            <div class="kpi-text">
            Average customer value
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # HIGHEST CLV
    with c3:

        max_clv_inr = usd_to_inr(
            df["CLV"].max()
        )

        st.markdown(
            f"""
            <div class="kpi">

            <div class="kpi-label">
            HIGHEST CLV
            </div>

            <div class="kpi-value">
            ₹{max_clv_inr:,.0f}
            </div>

            <div class="kpi-text">
            Maximum customer value
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ML ALGORITHM
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
    # PREDICTION ACTIVITY
    # ========================================================

    predicted_customers = df[
        df["Source"] == "Predicted"
    ]

    if len(predicted_customers) > 0:

        st.markdown(
            '<div class="section-title">Prediction Activity</div>',
            unsafe_allow_html=True
        )

        p1, p2, p3 = st.columns(3)

        with p1:

            st.metric(
                "Predicted Customers",
                len(predicted_customers)
            )

        with p2:

            latest_clv = predicted_customers["CLV"].iloc[-1]

            st.metric(
                "Latest Predicted CLV",
                format_inr(
                    usd_to_inr(latest_clv)
                )
            )

        with p3:

            total_predicted_value = predicted_customers["CLV"].sum()

            st.metric(
                "Total Predicted CLV",
                format_inr(
                    usd_to_inr(total_predicted_value)
                )
            )


    # ========================================================
    # CUSTOMER VALUE ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">Customer Value Analysis</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2)


    # --------------------------------------------------------
    # CLV DISTRIBUTION
    # --------------------------------------------------------

    with chart1:

        st.markdown("### 📈 CLV Distribution")

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        clv_inr = df["CLV"] * USD_TO_INR

        ax.hist(
            clv_inr,
            bins=min(10, max(5, len(df) // 2)),
            alpha=0.75
        )

        ax.set_xlabel(
            "Customer Lifetime Value (₹)"
        )

        ax.set_ylabel(
            "Number of Customers"
        )

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


    # --------------------------------------------------------
    # CUSTOMER CLV
    # --------------------------------------------------------

    with chart2:

        st.markdown("### 💰 Customer CLV")

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        customer_values = df["CLV"] * USD_TO_INR

        ax.bar(
            range(1, len(df) + 1),
            customer_values
        )

        ax.set_xlabel(
            "Customer Number"
        )

        ax.set_ylabel(
            "CLV (₹)"
        )

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
    # RECENT CUSTOMER RECORDS
    # ========================================================

    if len(predicted_customers) > 0:

        st.markdown(
            '<div class="section-title">Recently Predicted Customers</div>',
            unsafe_allow_html=True
        )

        recent = predicted_customers.tail(10).copy()

        recent["CLV_INR"] = (
            recent["CLV"] * USD_TO_INR
        )

        recent_display = recent[
            [
                "Customer_ID",
                "Recency",
                "Frequency",
                "Monetary",
                "CLV",
                "CLV_INR",
                "Source"
            ]
        ].copy()

        recent_display = recent_display.rename(
            columns={
                "Monetary": "Monetary_USD",
                "CLV": "CLV_USD"
            }
        )

        st.dataframe(
            recent_display,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # PLATFORM CAPABILITIES
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
            🇮🇳 INR Conversion
            </div>

            <div class="info-text">
            Predicted customer lifetime value is
            presented in Indian Rupees for easier
            business interpretation.
            </div>

            </div>
            """,
            unsafe_allow_html=True
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
        Every successful prediction is automatically added
        as a new customer record.
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
            "💰 Monetary Value (USD)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=50.0
        )


    # Show INR equivalent

    monetary_inr = usd_to_inr(
        monetary
    )

    st.info(
        f"💱 Monetary Value: ${monetary:,.2f} "
        f"≈ ₹{monetary_inr:,.2f}"
    )


    st.write("")


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "🚀 Predict Customer Lifetime Value",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Prepare user data
        # ----------------------------------------------------

        user_data = pd.DataFrame(
            {
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            }
        )


        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        prediction_usd = model.predict(
            user_data
        )[0]


        # Prevent negative CLV

        prediction_usd = max(
            0.0,
            float(prediction_usd)
        )


        prediction_inr = usd_to_inr(
            prediction_usd
        )


        # ====================================================
        # CREATE NEW CUSTOMER
        # ====================================================

        current_customer_count = len(
            st.session_state.customers
        )

        new_customer_number = (
            current_customer_count + 1
        )

        new_customer_id = (
            f"CUST-{new_customer_number:04d}"
        )


        new_customer = pd.DataFrame(
            {
                "Customer_ID": [
                    new_customer_id
                ],

                "Recency": [
                    recency
                ],

                "Frequency": [
                    frequency
                ],

                "Monetary": [
                    monetary
                ],

                "CLV": [
                    prediction_usd
                ],

                "Source": [
                    "Predicted"
                ]
            }
        )


        # ====================================================
        # ADD NEW CUSTOMER TO DATABASE
        # ====================================================

        st.session_state.customers = pd.concat(
            [
                st.session_state.customers,
                new_customer
            ],
            ignore_index=True
        )


        # Store prediction

        st.session_state.last_prediction = {
            "Customer_ID": new_customer_id,
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "CLV": prediction_usd
        }


        st.session_state.prediction_history.append(
            new_customer_id
        )


        # Update current dataframe

        df = st.session_state.customers.copy()


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            f"✅ Customer {new_customer_id} successfully added!"
        )


        st.info(
            f"👥 Total Customers: {len(df)} "
            f"(previously {len(df) - 1})"
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

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
            ₹{prediction_inr:,.2f}
            </div>

            <div class="prediction-usd">
            Original ML Prediction: ${prediction_usd:,.2f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        # ====================================================
        # CUSTOMER VALUE CATEGORY
        # ====================================================

        q75 = base_df["CLV"].quantile(
            0.75
        )

        median = base_df["CLV"].median()


        if prediction_usd >= q75:

            st.success(
                "⭐ HIGH-VALUE CUSTOMER\n\n"
                "Recommended for premium retention "
                "and loyalty campaigns."
            )

        elif prediction_usd >= median:

            st.info(
                "📈 MEDIUM-VALUE CUSTOMER\n\n"
                "Suitable for engagement and "
                "personalized campaigns."
            )

        else:

            st.warning(
                "📌 LOWER-VALUE CUSTOMER\n\n"
                "Consider targeted marketing strategies."
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.markdown(
            "### 📋 Prediction Summary"
        )


        result_table = pd.DataFrame(
            {
                "Parameter": [
                    "Customer ID",
                    "Recency",
                    "Frequency",
                    "Monetary Value",
                    "Predicted CLV",
                    "Total Customers"
                ],

                "Value": [
                    new_customer_id,
                    f"{recency} days",
                    frequency,
                    f"${monetary:,.2f} / ₹{monetary_inr:,.2f}",
                    f"₹{prediction_inr:,.2f}",
                    len(df)
                ]
            }
        )


        st.dataframe(
            result_table,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # NEW CUSTOMER RECORD
        # ====================================================

        st.markdown(
            "### 👤 New Customer Added"
        )


        display_new = new_customer.copy()

        display_new["CLV_INR"] = (
            display_new["CLV"] * USD_TO_INR
        )

        display_new = display_new.rename(
            columns={
                "Monetary": "Monetary_USD",
                "CLV": "CLV_USD"
            }
        )


        st.dataframe(
            display_new,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # NEXT CUSTOMER MESSAGE
        # ====================================================

        st.success(
            f"🎯 Ready for the next customer. "
            f"Current customer count: {len(df)}"
        )


# ============================================================
# SEGMENTATION DASHBOARD
# ============================================================

elif menu == "📊 Segmentation Dashboard":

    # Always use latest customer data

    df = st.session_state.customers.copy()


    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🎯 Customer Segmentation
        </div>

        <div class="hero-text">
        Discover customer groups using K-Means clustering.
        The segmentation uses all current customers,
        including every newly predicted customer.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.info(
        f"👥 Current customers available for segmentation: "
        f"**{len(df)}**"
    )


    # Need at least 2 customers for K-Means

    if len(df) < 2:

        st.warning(
            "At least 2 customers are required for segmentation."
        )

    else:

        max_clusters = min(
            12,
            len(df)
        )

        default_clusters = min(
            3,
            max_clusters
        )

        clusters = st.slider(
            "Number of Customer Segments",
            min_value=2,
            max_value=max_clusters,
            value=default_clusters
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
                kmeans.fit_predict(
                    scaled
                ) + 1
            )


            st.success(
                f"Successfully created {clusters} "
                f"customer segments from {len(df)} customers."
            )


            # ------------------------------------------------
            # CHART
            # ------------------------------------------------

            st.markdown(
                "### 📊 Customer Segment Map"
            )


            fig, ax = plt.subplots(
                figsize=(10, 5)
            )


            ax.scatter(
                segmented["Recency"],
                segmented["Monetary"] * USD_TO_INR,
                c=segmented["Cluster"],
                s=130,
                alpha=0.8
            )


            ax.set_xlabel(
                "Recency (Days)"
            )

            ax.set_ylabel(
                "Monetary Value (₹)"
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


            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.markdown(
                "### 📋 Segment Summary"
            )


            summary = (
                segmented
                .groupby("Cluster")
                .agg(
                    Customers=("CLV", "count"),

                    Average_CLV=(
                        "CLV",
                        "mean"
                    ),

                    Average_Frequency=(
                        "Frequency",
                        "mean"
                    ),

                    Average_Monetary=(
                        "Monetary",
                        "mean"
                    )
                )
                .reset_index()
            )


            # Convert CLV to INR

            summary["Average_CLV"] = (
                summary["Average_CLV"]
                * USD_TO_INR
            )


            summary["Average_Monetary"] = (
                summary["Average_Monetary"]
                * USD_TO_INR
            )


            summary = summary.rename(
                columns={
                    "Average_CLV":
                        "Average_CLV_INR",

                    "Average_Monetary":
                        "Average_Monetary_INR"
                }
            )


            st.dataframe(
                summary.style.format(
                    {
                        "Average_CLV_INR":
                            "₹{:,.2f}",

                        "Average_Frequency":
                            "{:.2f}",

                        "Average_Monetary_INR":
                            "₹{:,.2f}"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # FULL SEGMENTED CUSTOMER DATA
            # ------------------------------------------------

            st.markdown(
                "### 👥 All Customers by Segment"
            )


            segmented_display = segmented.copy()

            segmented_display["Monetary_INR"] = (
                segmented_display["Monetary"]
                * USD_TO_INR
            )

            segmented_display["CLV_INR"] = (
                segmented_display["CLV"]
                * USD_TO_INR
            )


            segmented_display = segmented_display.rename(
                columns={
                    "Monetary": "Monetary_USD",
                    "CLV": "CLV_USD"
                }
            )


            st.dataframe(
                segmented_display,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # SEGMENT INSIGHTS
            # ------------------------------------------------

            st.markdown(
                "### 💡 Segment Interpretation"
            )


            insight1, insight2, insight3 = st.columns(3)


            with insight1:

                st.info(
                    "⭐ **High-value segment**\n\n"
                    "Customers with higher purchase "
                    "frequency and monetary value."
                )


            with insight2:

                st.warning(
                    "📈 **Growth segment**\n\n"
                    "Customers who may have potential "
                    "for increased engagement."
                )


            with insight3:

                st.success(
                    "🎯 **Business Strategy**\n\n"
                    "Use segments for personalized "
                    "marketing and retention campaigns."
                )


# ============================================================
# ADMIN PANEL
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


    # ========================================================
    # BLOCKED
    # ========================================================

    if st.session_state.blocked:

        st.error(
            "🚫 Access blocked after 3 incorrect attempts."
        )

        st.info(
            "Please contact the administrator for access."
        )

        st.stop()


    # ========================================================
    # LOGIN
    # ========================================================

    if not st.session_state.admin_logged_in:

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

                st.session_state.admin_logged_in = True
                st.session_state.attempts = 0

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
                        f"❌ Wrong password! "
                        f"Attempts remaining: {remaining}"
                    )

                else:

                    st.session_state.blocked = True

                    st.error(
                        "🚫 You are blocked after "
                        "3 wrong attempts."
                    )

                    st.rerun()


    # ========================================================
    # ADMIN DASHBOARD
    # ========================================================

    else:

        df = st.session_state.customers.copy()


        st.success(
            "✅ Administrator Access Active"
        )


        if st.button(
            "🔒 Logout",
            use_container_width=False
        ):

            st.session_state.admin_logged_in = False

            st.rerun()


        st.markdown(
            "### 📊 Admin Dashboard"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Customers",
                len(df)
            )


        with c2:

            st.metric(
                "Predicted Customers",
                len(
                    df[df["Source"] == "Predicted"]
                )
            )


        with c3:

            st.metric(
                "Average CLV",
                format_inr(
                    usd_to_inr(
                        df["CLV"].mean()
                    )
                )
            )


        with c4:

            st.metric(
                "Maximum CLV",
                format_inr(
                    usd_to_inr(
                        df["CLV"].max()
                    )
                )
            )


        st.divider()


        # ====================================================
        # CUSTOMER RECORDS
        # ====================================================

        st.markdown(
            "### 📄 Customer Records"
        )


        admin_df = df.copy()


        admin_df["CLV_INR"] = (
            admin_df["CLV"]
            * USD_TO_INR
        )


        admin_df["Monetary_INR"] = (
            admin_df["Monetary"]
            * USD_TO_INR
        )


        admin_df = admin_df.rename(
            columns={
                "CLV": "CLV_USD",
                "Monetary": "Monetary_USD"
            }
        )


        st.dataframe(
            admin_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DOWNLOAD CSV
        # ====================================================

        csv = admin_df.to_csv(
            index=False
        )


        st.download_button(
            "📥 Download Customer Data",
            csv,
            "customer_clv_data.csv",
            "text/csv",
            use_container_width=True
        )


        # ====================================================
        # PREDICTED CUSTOMER COUNT
        # ====================================================

        predicted_df = df[
            df["Source"] == "Predicted"
        ]


        if len(predicted_df) > 0:

            st.divider()

            st.markdown(
                "### 🔮 Predicted Customer Records"
            )


            st.success(
                f"{len(predicted_df)} newly predicted "
                f"customers are currently stored."
            )


            st.dataframe(
                predicted_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.markdown(
    """
    <div class="footer">

    📊 Customer Lifetime Value Prediction
    &nbsp; • &nbsp;
    Machine Learning Project
    &nbsp; • &nbsp;
    🇮🇳 Indian Rupee (INR)
    &nbsp; • &nbsp;
    Python + Scikit-learn + Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
