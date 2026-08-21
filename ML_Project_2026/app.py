# ============================================================
# CUSTOMER LIFETIME VALUE - ML DASHBOARD
# Clean Streamlit UI - NO HTML
# NO DATASET UPLOAD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# BUILT-IN DATASET
# ============================================================

@st.cache_data
def load_data():

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

    return pd.DataFrame(data)


df = load_data()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    X = data[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ]

    y = data["CLV"]

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

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        mae,
        rmse,
        r2
    )


(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    predictions,
    mae,
    rmse,
    r2
) = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 CLV Intelligence")

    st.caption(
        "Customer Lifetime Value Analytics"
    )

    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 CLV Predictor",
            "🎯 Customer Segmentation",
            "🤖 ML Model",
            "🔐 Admin Panel",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.success("🟢 Model Ready")

    st.caption("Machine Learning")
    st.write("Gradient Boosting")

    st.caption("Segmentation")
    st.write("K-Means")

    st.caption("Dataset")
    st.write("Built-in CLV Dataset")

    st.divider()

    st.caption(
        "Python • Pandas • Scikit-learn • Streamlit"
    )


# ============================================================
# TOP HEADER
# ============================================================

st.title("Customer Lifetime Value")

st.caption(
    "Machine Learning powered customer intelligence platform"
)


# ============================================================
# DASHBOARD
# ============================================================

if menu == "🏠 Dashboard":

    st.subheader("📈 Business Overview")

    total_customers = len(df)
    average_clv = df["CLV"].mean()
    maximum_clv = df["CLV"].max()
    median_clv = df["CLV"].median()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with c2:
        st.metric(
            "💰 Average CLV",
            f"${average_clv:,.0f}"
        )

    with c3:
        st.metric(
            "🏆 Highest CLV",
            f"${maximum_clv:,.0f}"
        )

    with c4:
        st.metric(
            "📊 Median CLV",
            f"${median_clv:,.0f}"
        )

    st.divider()

    # --------------------------------------------------------
    # PROJECT INTRO
    # --------------------------------------------------------

    st.subheader("🚀 Project Overview")

    left, right = st.columns([2, 1])

    with left:

        st.info(
            """
            **Customer Lifetime Value (CLV)** is a machine
            learning application that estimates the future
            value of customers.

            The system uses customer purchasing behavior:

            • Recency  
            • Frequency  
            • Monetary Value  

            Gradient Boosting is used for CLV prediction,
            while K-Means clustering identifies customer
            segments.
            """
        )

    with right:

        st.metric(
            "🤖 ML Algorithm",
            "Gradient Boosting"
        )

        st.metric(
            "🎯 Clustering",
            "K-Means"
        )

    st.divider()

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    st.subheader("📊 Customer Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### CLV Distribution")

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.hist(
            df["CLV"],
            bins=7,
            alpha=0.75
        )

        ax.set_xlabel(
            "Customer Lifetime Value"
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

        plt.close(fig)

    with col2:

        st.write("### Customer CLV")

        chart_data = df[
            ["CLV"]
        ].copy()

        chart_data.index = [
            f"Customer {i}"
            for i in range(
                1,
                len(df) + 1
            )
        ]

        st.bar_chart(
            chart_data
        )

    st.divider()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader("📋 Customer Data")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CLV PREDICTOR
# ============================================================

elif menu == "🔮 CLV Predictor":

    st.subheader(
        "🔮 Customer Lifetime Value Predictor"
    )

    st.write(
        "Enter customer purchasing behavior to estimate "
        "their future lifetime value."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        recency = st.number_input(
            "🕒 Recency (Days)",
            min_value=0,
            max_value=365,
            value=30,
            step=1
        )

    with col2:

        frequency = st.number_input(
            "🔄 Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )

    with col3:

        monetary = st.number_input(
            "💰 Monetary Value",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=50.0
        )

    st.write("")

    predict = st.button(
        "🚀 Predict Customer CLV",
        use_container_width=True
    )

    if predict:

        user_data = pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary]
        })

        prediction = model.predict(
            user_data
        )[0]

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted CLV",
                f"${prediction:,.2f}"
            )

        with result_col2:

            if prediction >= df["CLV"].quantile(0.75):

                st.success(
                    "⭐ HIGH-VALUE CUSTOMER"
                )

            elif prediction >= df["CLV"].median():

                st.info(
                    "📈 MEDIUM-VALUE CUSTOMER"
                )

            else:

                st.warning(
                    "📌 LOWER-VALUE CUSTOMER"
                )

        st.divider()

        st.subheader(
            "Customer Profile"
        )

        profile = pd.DataFrame({
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
            profile,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "💡 Business Insight: High-value customers "
            "can be prioritized for retention, loyalty "
            "programs and personalized offers."
        )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif menu == "🎯 Customer Segmentation":

    st.subheader(
        "🎯 Customer Segmentation"
    )

    st.write(
        "K-Means clustering groups customers according "
        "to their purchasing behavior."
    )

    st.divider()

    number_of_clusters = st.slider(
        "Number of Customer Segments",
        min_value=2,
        max_value=5,
        value=3
    )

    run_segmentation = st.button(
        "🚀 Generate Customer Segments",
        use_container_width=True
    )

    if run_segmentation:

        features = df[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(
            features
        )

        kmeans = KMeans(
            n_clusters=number_of_clusters,
            random_state=42,
            n_init=10
        )

        segmented_df = df.copy()

        segmented_df["Segment"] = (
            kmeans.fit_predict(
                scaled_data
            ) + 1
        )

        st.success(
            f"Successfully created "
            f"{number_of_clusters} customer segments."
        )

        st.divider()

        # ----------------------------------------------------
        # SEGMENT COUNTS
        # ----------------------------------------------------

        st.subheader(
            "👥 Segment Distribution"
        )

        segment_counts = (
            segmented_df["Segment"]
            .value_counts()
            .sort_index()
        )

        chart_data = pd.DataFrame({
            "Customers": segment_counts
        })

        st.bar_chart(
            chart_data
        )

        st.divider()

        # ----------------------------------------------------
        # SEGMENT TABLE
        # ----------------------------------------------------

        st.subheader(
            "📊 Segment Performance"
        )

        summary = (
            segmented_df
            .groupby("Segment")
            .agg(
                Customers=("CLV", "count"),
                Average_CLV=("CLV", "mean"),
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

        st.dataframe(
            summary.style.format({
                "Average_CLV": "${:,.2f}",
                "Average_Frequency": "{:.2f}",
                "Average_Monetary": "${:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        st.subheader(
            "📍 Customer Segment Map"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.scatter(
            segmented_df["Recency"],
            segmented_df["Monetary"],
            c=segmented_df["Segment"],
            s=120,
            alpha=0.8
        )

        ax.set_xlabel(
            "Recency"
        )

        ax.set_ylabel(
            "Monetary"
        )

        ax.set_title(
            "K-Means Customer Segmentation"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.subheader(
            "📋 Segmented Customers"
        )

        st.dataframe(
            segmented_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ML MODEL
# ============================================================

elif menu == "🤖 ML Model":

    st.subheader(
        "🤖 Machine Learning Model"
    )

    st.write(
        "Model performance and technical information."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    with c2:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

    with c3:

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    st.subheader(
        "🧠 Model Information"
    )

    info1, info2 = st.columns(2)

    with info1:

        st.info(
            """
            **Algorithm**

            Gradient Boosting Regressor

            **Problem Type**

            Supervised Machine Learning

            **Task**

            Regression

            **Target**

            Customer Lifetime Value
            """
        )

    with info2:

        st.info(
            """
            **Input Features**

            • Recency  
            • Frequency  
            • Monetary

            **Training**

            80% Training Data

            **Testing**

            20% Testing Data
            """
        )

    st.divider()

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader(
        "📈 Actual vs Predicted CLV"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.scatter(
        y_test,
        predictions,
        s=80,
        alpha=0.75
    )

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel(
        "Actual CLV"
    )

    ax.set_ylabel(
        "Predicted CLV"
    )

    ax.set_title(
        "Actual vs Predicted Customer Lifetime Value"
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
    # FEATURES
    # --------------------------------------------------------

    st.subheader(
        "🔍 Model Features"
    )

    feature_df = pd.DataFrame({
        "Feature": [
            "Recency",
            "Frequency",
            "Monetary"
        ],

        "Description": [
            "Days since last purchase",
            "Number of purchases",
            "Total customer spending"
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ADMIN PANEL
# ============================================================

elif menu == "🔐 Admin Panel":

    st.subheader(
        "🔐 Administrator Panel"
    )

    st.write(
        "Authorized access only."
    )

    st.divider()

    if "attempts" not in st.session_state:

        st.session_state.attempts = 0

    if "blocked" not in st.session_state:

        st.session_state.blocked = False

    if st.session_state.blocked:

        st.error(
            "🚫 Access blocked after 3 incorrect attempts."
        )

        st.info(
            "Please contact the project administrator."
        )

        st.stop()

    password = st.text_input(
        "🔑 Administrator Password",
        type="password"
    )

    login = st.button(
        "🔓 Login",
        use_container_width=True
    )

    if login:

        if password.strip() == "admin123":

            st.session_state.attempts = 0

            st.success(
                "✅ Access Granted"
            )

            st.divider()

            st.subheader(
                "📊 Admin Dashboard"
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

            st.subheader(
                "📋 Customer Records"
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
                file_name="customer_clv_data.csv",
                mime="text/csv",
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
                    f"❌ Incorrect password. "
                    f"Attempts remaining: {remaining}"
                )

            else:

                st.session_state.blocked = True

                st.error(
                    "🚫 Too many incorrect attempts."
                )

                st.rerun()


# ============================================================
# ABOUT
# ============================================================

elif menu == "ℹ️ About":

    st.subheader(
        "ℹ️ About the Project"
    )

    st.write(
        """
        ### Customer Lifetime Value Prediction

        This project uses Machine Learning to estimate
        the potential lifetime value of customers.

        The application analyzes three important customer
        behavior features:

        **Recency** — how recently the customer purchased.

        **Frequency** — how often the customer purchases.

        **Monetary Value** — how much the customer spends.

        A Gradient Boosting Regressor is used to predict CLV.

        K-Means clustering is additionally used to divide
        customers into behavioral segments.
        """
    )

    st.divider()

    st.subheader(
        "🔄 Project Workflow"
    )

    steps = [
        "1️⃣ Customer Data",
        "2️⃣ Data Preparation",
        "3️⃣ Feature Selection",
        "4️⃣ ML Model Training",
        "5️⃣ CLV Prediction",
        "6️⃣ Customer Segmentation",
        "7️⃣ Business Insights"
    ]

    for step in steps:

        st.write(step)

    st.divider()

    st.subheader(
        "🛠 Technologies Used"
    )

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Streamlit"
        ],

        "Purpose": [
            "Programming",
            "Data Processing",
            "Numerical Computing",
            "Machine Learning",
            "Visualization",
            "Web Application"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.success(
        "🎯 Project Goal: Use Machine Learning to "
        "understand customer value and support better "
        "business decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📊 Customer Lifetime Value Prediction "
    "• Machine Learning Project "
    "• Built with Python & Streamlit"
)
