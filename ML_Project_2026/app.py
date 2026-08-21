# ============================================================
# CUSTOMER LIFETIME VALUE - ML DASHBOARD
# Complete Streamlit Application
# Automatic Dataset Detection
# No Upload / No Dataset Selector
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Lifetime Value",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.70;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.06);
        min-height: 130px;
        transition: 0.2s;
    }

    .metric-card:hover {
        border-color: rgba(128,128,128,0.35);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 14px;
        opacity: 0.65;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 750;
    }

    .metric-description {
        font-size: 12px;
        opacity: 0.55;
        margin-top: 6px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.05);
        margin-bottom: 12px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.15);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
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
# AUTOMATIC DATASET LOADING
# ============================================================

@st.cache_data
def load_dataset():

    # Location of this app.py
    app_directory = Path(__file__).resolve().parent

    # Search every CSV in this project and subfolders
    csv_files = list(
        app_directory.rglob("*.csv")
    )

    # Ignore hidden folders/files
    csv_files = [
        file
        for file in csv_files
        if not any(
            part.startswith(".")
            for part in file.parts
        )
    ]

    # --------------------------------------------------------
    # No CSV found
    # --------------------------------------------------------

    if not csv_files:

        st.error(
            "❌ Dataset not found."
        )

        st.info(
            "Make sure your CSV dataset is inside the "
            "GitHub project and committed to the repository."
        )

        st.stop()

    # --------------------------------------------------------
    # Preferred CLV filenames
    # --------------------------------------------------------

    preferred_names = [
        "customer_clv.csv",
        "clv.csv",
        "customer_lifetime_value.csv",
        "customer_lifetime_value_dataset.csv",
        "customers.csv",
        "customer_data.csv",
        "data.csv",
        "dataset.csv"
    ]

    selected_file = None

    # First search for known filenames
    for preferred_name in preferred_names:

        for file in csv_files:

            if file.name.lower() == preferred_name.lower():

                selected_file = file

                break

        if selected_file is not None:
            break

    # --------------------------------------------------------
    # Otherwise automatically use first CSV
    # --------------------------------------------------------

    if selected_file is None:

        selected_file = csv_files[0]

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        data = pd.read_csv(
            selected_file
        )

        return data

    except Exception:

        st.error(
            "❌ Dataset was found but could not be read."
        )

        st.stop()


# Load automatically
df = load_dataset()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(dataframe, possible_names):

    columns_lower = {
        str(col).lower().strip(): col
        for col in dataframe.columns
    }

    # Exact match
    for name in possible_names:

        if name.lower() in columns_lower:

            return columns_lower[
                name.lower()
            ]

    # Partial match
    for col in dataframe.columns:

        col_lower = str(
            col
        ).lower().strip()

        for name in possible_names:

            if name.lower() in col_lower:

                return col

    return None


def numeric_columns(dataframe):

    return dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()


def prepare_data(dataframe):

    data = dataframe.copy()

    # Remove empty columns
    data = data.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicates
    data = data.drop_duplicates()

    # Convert numeric-looking object columns
    for col in data.columns:

        if data[col].dtype == "object":

            converted = pd.to_numeric(
                data[col],
                errors="coerce"
            )

            valid_ratio = (
                converted.notna().mean()
            )

            if valid_ratio > 0.80:

                data[col] = converted

    return data


def create_target(dataframe):

    target_candidates = [

        "clv",

        "customer_lifetime_value",

        "customer lifetime value",

        "life_time_value",

        "lifetime_value",

        "lifetime value",

        "customer_value",

        "customer value",

        "monetary_value"

    ]

    return find_column(
        dataframe,
        target_candidates
    )


def create_features(dataframe, target):

    data = dataframe.copy()

    numerical = data.select_dtypes(
        include=np.number
    ).copy()

    # Remove CLV target
    if target in numerical.columns:

        numerical = numerical.drop(
            columns=[target]
        )

    # Remove ID columns
    remove_columns = []

    for col in numerical.columns:

        name = str(
            col
        ).lower()

        if (
            name == "id"
            or name.endswith("_id")
            or name.endswith(" id")
            or "customerid" in name.replace(
                "_",
                ""
            )
        ):

            remove_columns.append(col)

    numerical = numerical.drop(
        columns=remove_columns,
        errors="ignore"
    )

    # Replace infinite values
    numerical = numerical.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    numerical = numerical.fillna(
        numerical.median(
            numeric_only=True
        )
    )

    numerical = numerical.fillna(0)

    return numerical


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(dataframe, target):

    X = create_features(
        dataframe,
        target
    )

    y = pd.to_numeric(
        dataframe[target],
        errors="coerce"
    )

    # Remove rows with invalid target
    valid = y.notna()

    X = X.loc[valid]

    y = y.loc[valid]

    # Minimum data requirement
    if len(X) < 10:

        return None

    if X.shape[1] == 0:

        return None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42
    )

    # Gradient Boosting
    model = GradientBoostingRegressor(

        n_estimators=150,

        learning_rate=0.05,

        max_depth=3,

        random_state=42,

        loss="squared_error"
    )

    model.fit(
        X_train,
        y_train
    )

    # Prediction
    predictions = model.predict(
        X_test
    )

    # Metrics
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

    return {

        "model": model,

        "features": X.columns.tolist(),

        "X": X,

        "y": y,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "predictions": predictions,

        "mae": mae,

        "rmse": rmse,

        "r2": r2
    }


# ============================================================
# PREPARE DATA
# ============================================================

df = prepare_data(
    df
)

target_column = create_target(
    df
)


# Train model automatically
if target_column is not None:

    result = train_model(
        df,
        target_column
    )

else:

    result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 📊 CLV Intelligence"
    )

    st.caption(
        "Machine Learning Customer Analytics"
    )

    st.divider()

    page = st.radio(

        "Navigation",

        [

            "🏠 Dashboard",

            "👤 Customer Prediction",

            "📊 Analytics",

            "🎯 Customer Segments",

            "🤖 ML Model",

            "ℹ️ About Project"

        ]
    )

    st.divider()

    # System status
    st.success(
        "● ML System Online"
    )

    st.caption(
        "Customer data connected automatically"
    )

    st.caption(
        "Gradient Boosting + K-Means"
    )

    st.divider()

    st.caption(
        "Built with Python • "
        "Scikit-learn • Streamlit"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        Customer Lifetime Value
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Machine Learning powered customer intelligence dashboard
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="section-title">
            Business Overview
        </div>
        """,
        unsafe_allow_html=True
    )

    if target_column is not None:

        clv_values = pd.to_numeric(

            df[target_column],

            errors="coerce"

        ).dropna()

        total_customers = len(
            df
        )

        average_clv = clv_values.mean()

        high_value_limit = (
            clv_values.quantile(0.75)
        )

        high_value_customers = (
            clv_values >= high_value_limit
        ).sum()

        maximum_clv = clv_values.max()

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        TOTAL CUSTOMERS
                    </div>

                    <div class="metric-value">
                        {total_customers:,}
                    </div>

                    <div class="metric-description">
                        Customers analyzed
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        AVERAGE CLV
                    </div>

                    <div class="metric-value">
                        {average_clv:,.2f}
                    </div>

                    <div class="metric-description">
                        Average customer value
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        HIGH-VALUE CUSTOMERS
                    </div>

                    <div class="metric-value">
                        {high_value_customers:,}
                    </div>

                    <div class="metric-description">
                        Top 25% by CLV
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        MAXIMUM CLV
                    </div>

                    <div class="metric-value">
                        {maximum_clv:,.2f}
                    </div>

                    <div class="metric-description">
                        Highest customer value
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 📈 CLV Distribution"
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.hist(
                clv_values,
                bins=25
            )

            ax.set_xlabel(
                "Customer Lifetime Value"
            )

            ax.set_ylabel(
                "Number of Customers"
            )

            ax.set_title(
                "Customer Value Distribution"
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

            st.markdown(
                "### 📊 Customer Value Summary"
            )

            summary = pd.DataFrame({

                "Metric": [

                    "Minimum CLV",

                    "Average CLV",

                    "Median CLV",

                    "Maximum CLV"

                ],

                "Value": [

                    clv_values.min(),

                    clv_values.mean(),

                    clv_values.median(),

                    clv_values.max()

                ]

            })

            st.dataframe(

                summary.style.format({

                    "Value": "{:,.2f}"

                }),

                use_container_width=True,

                hide_index=True

            )

            st.markdown(
                """
                <div class="info-box">

                <b>💡 Business Insight</b>

                <br><br>

                Customers with higher lifetime value
                can be prioritized for retention campaigns,
                personalized offers and loyalty programs.

                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                Customer Data Preview
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(

            df.head(10),

            use_container_width=True,

            hide_index=True
        )

    else:

        st.error(
            "CLV target column could not be detected."
        )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "👤 Customer Prediction":

    st.markdown(
        """
        <div class="section-title">
            Customer CLV Prediction
        </div>
        """,
        unsafe_allow_html=True
    )

    if result is None:

        st.error(
            "The ML model could not be trained."
        )

    else:

        st.info(
            "Enter customer information to estimate "
            "Customer Lifetime Value."
        )

        features = result["features"]

        model = result["model"]

        input_values = {}

        columns = st.columns(2)

        for index, feature in enumerate(
            features
        ):

            with columns[index % 2]:

                series = pd.to_numeric(

                    df[feature],

                    errors="coerce"

                )

                default_value = float(
                    series.median()
                )

                minimum = float(
                    series.min()
                )

                maximum = float(
                    series.max()
                )

                if not np.isfinite(
                    default_value
                ):

                    default_value = 0

                if not np.isfinite(
                    minimum
                ):

                    minimum = 0

                if not np.isfinite(
                    maximum
                ):

                    maximum = max(
                        default_value + 1,
                        1
                    )

                if minimum == maximum:

                    maximum = minimum + 1

                input_values[feature] = st.number_input(

                    feature.replace(
                        "_",
                        " "
                    ).title(),

                    min_value=minimum,

                    max_value=maximum,

                    value=min(

                        max(
                            default_value,
                            minimum
                        ),

                        maximum

                    )
                )

        st.write("")

        if st.button(

            "🔮 Predict Customer Lifetime Value",

            use_container_width=True

        ):

            input_df = pd.DataFrame(
                [input_values]
            )

            prediction = model.predict(
                input_df
            )[0]

            st.success(
                "Prediction completed successfully."
            )

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        PREDICTED CUSTOMER LIFETIME VALUE
                    </div>

                    <div class="metric-value">
                        {prediction:,.2f}
                    </div>

                    <div class="metric-description">
                        Machine Learning prediction
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            q75 = df[
                target_column
            ].quantile(
                0.75
            )

            median = df[
                target_column
            ].median()

            if prediction >= q75:

                st.success(
                    "⭐ High-value customer — "
                    "strong retention priority."
                )

            elif prediction >= median:

                st.info(
                    "📈 Medium-value customer — "
                    "suitable for engagement campaigns."
                )

            else:

                st.warning(
                    "📌 Lower-value customer — "
                    "consider targeted marketing."
                )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        """
        <div class="section-title">
            Customer Analytics
        </div>
        """,
        unsafe_allow_html=True
    )

    numerical = numeric_columns(
        df
    )

    if len(numerical) >= 2:

        col1, col2 = st.columns(2)

        with col1:

            x_column = st.selectbox(
                "Select X-axis",
                numerical
            )

        with col2:

            y_options = [

                col

                for col in numerical

                if col != x_column

            ]

            y_column = st.selectbox(
                "Select Y-axis",
                y_options
            )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(

            df[x_column],

            df[y_column],

            alpha=0.55
        )

        ax.set_xlabel(
            x_column
        )

        ax.set_ylabel(
            y_column
        )

        ax.set_title(
            f"{x_column} vs {y_column}"
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
    # CORRELATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Correlation Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    correlation = df[
        numerical
    ].corr()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    image = ax.imshow(

        correlation,

        aspect="auto"
    )

    ax.set_xticks(
        range(
            len(
                correlation.columns
            )
        )
    )

    ax.set_yticks(
        range(
            len(
                correlation.columns
            )
        )
    )

    ax.set_xticklabels(

        correlation.columns,

        rotation=45,

        ha="right"
    )

    ax.set_yticklabels(
        correlation.columns
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_title(
        "Feature Correlation Matrix"
    )

    st.pyplot(

        fig,

        use_container_width=True

    )

    plt.close(fig)


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

elif page == "🎯 Customer Segments":

    st.markdown(
        """
        <div class="section-title">
            Customer Segmentation
        </div>
        """,
        unsafe_allow_html=True
    )

    numerical = numeric_columns(
        df
    )

    if len(numerical) >= 2:

        segment_data = df[
            numerical
        ].copy()

        segment_data = segment_data.replace(

            [np.inf, -np.inf],

            np.nan
        )

        segment_data = segment_data.fillna(

            segment_data.median()
        )

        segment_data = segment_data.fillna(
            0
        )

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            segment_data
        )

        number_of_clusters = st.slider(

            "Number of Customer Segments",

            min_value=2,

            max_value=6,

            value=4
        )

        kmeans = KMeans(

            n_clusters=number_of_clusters,

            random_state=42,

            n_init=10
        )

        clusters = kmeans.fit_predict(
            scaled
        )

        segmented_df = df.copy()

        segmented_df[
            "Customer Segment"
        ] = clusters + 1

        st.success(
            f"{number_of_clusters} customer segments "
            "created successfully."
        )

        segment_counts = (

            segmented_df[
                "Customer Segment"
            ]

            .value_counts()

            .sort_index()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 👥 Segment Distribution"
            )

            fig, ax = plt.subplots(
                figsize=(7, 4)
            )

            ax.bar(

                segment_counts.index.astype(str),

                segment_counts.values

            )

            ax.set_xlabel(
                "Customer Segment"
            )

            ax.set_ylabel(
                "Customers"
            )

            ax.set_title(
                "Customers by Segment"
            )

            st.pyplot(

                fig,

                use_container_width=True

            )

            plt.close(fig)

        with col2:

            st.markdown(
                "### 📋 Segment Summary"
            )

            summary = (

                segmented_df

                .groupby(
                    "Customer Segment"
                )

                .size()

                .reset_index(
                    name="Customers"
                )

            )

            summary[
                "Percentage"
            ] = (

                summary["Customers"]

                / len(
                    segmented_df
                )

                * 100

            )

            st.dataframe(

                summary.style.format({

                    "Percentage": "{:.1f}%"

                }),

                use_container_width=True,

                hide_index=True

            )

        st.markdown(
            """
            <div class="section-title">
                Segmented Customers
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(

            segmented_df.head(20),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.warning(
            "At least two numerical columns are required "
            "for segmentation."
        )


# ============================================================
# ML MODEL
# ============================================================

elif page == "🤖 ML Model":

    st.markdown(
        """
        <div class="section-title">
            Machine Learning Model
        </div>
        """,
        unsafe_allow_html=True
    )

    if result is None:

        st.error(
            "Model training could not be completed."
        )

    else:

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "R² Score",

                f"{result['r2']:.3f}"

            )

        with col2:

            st.metric(

                "MAE",

                f"{result['mae']:.2f}"

            )

        with col3:

            st.metric(

                "RMSE",

                f"{result['rmse']:.2f}"

            )

        st.write("")

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="info-box">

            <b>🤖 Algorithm</b><br>
            Gradient Boosting Regressor

            </div>

            <div class="info-box">

            <b>📚 Problem Type</b><br>
            Supervised Machine Learning • Regression

            </div>

            <div class="info-box">

            <b>🎯 Objective</b><br>
            Predict the expected Customer Lifetime Value.

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        st.markdown(
            "### 🔍 Model Features"
        )

        feature_table = pd.DataFrame({

            "Feature":
                result["features"]

        })

        st.dataframe(

            feature_table,

            use_container_width=True,

            hide_index=True

        )

        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Actual vs Predicted CLV"
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.scatter(

            result["y_test"],

            result["predictions"],

            alpha=0.6

        )

        min_value = min(

            result["y_test"].min(),

            result["predictions"].min()

        )

        max_value = max(

            result["y_test"].max(),

            result["predictions"].max()

        )

        ax.plot(

            [min_value, max_value],

            [min_value, max_value],

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


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        """
        <div class="section-title">
            About the Project
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### 📊 Customer Lifetime Value Prediction

        Customer Lifetime Value Prediction is a Machine Learning
        application designed to estimate the future value of customers.

        The system analyzes customer-related information and uses a
        Gradient Boosting Regression model to predict Customer
        Lifetime Value.

        K-Means clustering is also used to identify different
        customer segments.

        ### 🔄 Project Workflow

        **Data Collection**

        ↓

        **Data Cleaning**

        ↓

        **Feature Preparation**

        ↓

        **Machine Learning**

        ↓

        **CLV Prediction**

        ↓

        **Customer Segmentation**

        ↓

        **Business Insights**

        ### 🧠 Technologies Used

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Matplotlib
        - Streamlit

        ### 🎯 Business Benefits

        - Identify high-value customers
        - Improve customer retention
        - Support targeted marketing
        - Understand customer segments
        - Make data-driven decisions
        """
    )

    st.divider()

    st.markdown(
        "### 📌 Project Information"
    )

    info = pd.DataFrame({

        "Component": [

            "Frontend",

            "Programming Language",

            "ML Algorithm",

            "Clustering",

            "Visualization",

            "Application Type"

        ],

        "Technology": [

            "Streamlit",

            "Python",

            "Gradient Boosting Regressor",

            "K-Means",

            "Matplotlib",

            "Machine Learning Web Application"

        ]

    })

    st.dataframe(

        info,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Lifetime Value Prediction • "
    "Machine Learning Project • "
    "Built with Python & Streamlit"
)
