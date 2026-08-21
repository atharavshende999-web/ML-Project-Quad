import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CLV Studio | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #e6e9ef;
}

[data-testid="stMetric"] {
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 16px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    font-size: 13px;
}

[data-testid="stMetricValue"] {
    font-size: 25px;
}

h1 {
    font-weight: 700;
    letter-spacing: -1px;
}

h2 {
    font-weight: 650;
}

h3 {
    font-weight: 600;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

div[data-baseweb="select"] > div {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

DATA_FOLDER = "data"

SUPPORTED_FILES = [
    ".csv",
    ".xlsx",
    ".xls"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_dataset():
    """
    Automatically searches common locations for a dataset.
    """

    possible_paths = [
        "data.csv",
        "dataset.csv",
        "customer_data.csv",
        "clv_data.csv",
        "customers.csv",
        "data/data.csv",
        "data/dataset.csv",
        "data/customer_data.csv",
        "data/clv_data.csv",
        "data/customers.csv",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    if os.path.exists(DATA_FOLDER):
        for file in os.listdir(DATA_FOLDER):
            if os.path.splitext(file)[1].lower() in SUPPORTED_FILES:
                return os.path.join(DATA_FOLDER, file)

    return None


@st.cache_data
def load_data(path):
    extension = os.path.splitext(path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    return None


def clean_dataframe(df):

    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Strip column names
    df.columns = [
        str(col).strip().replace("\n", " ")
        for col in df.columns
    ]

    return df


def find_column(columns, keywords):

    lower_columns = {
        str(col).lower(): col
        for col in columns
    }

    for keyword in keywords:

        for lower_col, original_col in lower_columns.items():

            if keyword in lower_col:
                return original_col

    return None


def detect_columns(df):

    columns = list(df.columns)

    customer_id = find_column(
        columns,
        [
            "customer_id",
            "customerid",
            "customer id",
            "user_id",
            "userid",
            "client_id",
            "id"
        ]
    )

    recency = find_column(
        columns,
        [
            "recency",
            "days_since",
            "days since",
            "last_purchase_days",
            "last purchase"
        ]
    )

    frequency = find_column(
        columns,
        [
            "frequency",
            "purchase_frequency",
            "purchase frequency",
            "orders",
            "transactions",
            "number_of_orders",
            "num_orders"
        ]
    )

    monetary = find_column(
        columns,
        [
            "monetary",
            "monetary_value",
            "monetary value",
            "revenue",
            "sales",
            "spend",
            "total_spend",
            "total spend",
            "amount"
        ]
    )

    clv = find_column(
        columns,
        [
            "clv",
            "customer_lifetime_value",
            "customer lifetime value",
            "lifetime_value",
            "lifetime value"
        ]
    )

    return {
        "customer_id": customer_id,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "clv": clv
    }


def numeric_columns(df):

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def create_clv_features(df, detected):

    work = df.copy()

    numeric = numeric_columns(work)

    # Recency
    if detected["recency"] is not None:

        work["CLV_Recency"] = pd.to_numeric(
            work[detected["recency"]],
            errors="coerce"
        )

    elif numeric:

        work["CLV_Recency"] = 0

    # Frequency
    if detected["frequency"] is not None:

        work["CLV_Frequency"] = pd.to_numeric(
            work[detected["frequency"]],
            errors="coerce"
        )

    elif len(numeric) >= 1:

        work["CLV_Frequency"] = pd.to_numeric(
            work[numeric[0]],
            errors="coerce"
        )

    # Monetary
    if detected["monetary"] is not None:

        work["CLV_Monetary"] = pd.to_numeric(
            work[detected["monetary"]],
            errors="coerce"
        )

    elif len(numeric) >= 2:

        work["CLV_Monetary"] = pd.to_numeric(
            work[numeric[1]],
            errors="coerce"
        )

    return work


def train_model(df, detected):

    features = [
        "CLV_Recency",
        "CLV_Frequency",
        "CLV_Monetary"
    ]

    available_features = [
        col for col in features
        if col in df.columns
    ]

    if len(available_features) < 2:
        return None

    work = df[
        available_features
    ].copy()

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    work = work.dropna()

    if len(work) < 10:
        return None

    # If actual CLV exists, use it
    if detected["clv"] is not None:

        target = pd.to_numeric(
            df.loc[work.index, detected["clv"]],
            errors="coerce"
        )

        valid = target.notna()

        X = work.loc[valid]
        y = target.loc[valid]

    else:

        # Estimated target when CLV is not available.
        # This keeps the dashboard functional with
        # common customer transaction datasets.
        recency = work["CLV_Recency"].clip(lower=0)

        frequency = work["CLV_Frequency"].clip(lower=0)

        monetary = work["CLV_Monetary"].clip(lower=0)

        y = (
            monetary
            * (1 + frequency)
            / (1 + recency)
        )

        X = work

    if len(X) < 10:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
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

    return {
        "model": model,
        "features": available_features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def create_segments(df):

    features = [
        "CLV_Recency",
        "CLV_Frequency",
        "CLV_Monetary"
    ]

    available = [
        f for f in features
        if f in df.columns
    ]

    if len(available) < 2:
        return df, None

    work = df[available].copy()

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    work = work.fillna(
        work.median(numeric_only=True)
    )

    if len(work) < 4:
        return df, None

    scaler = StandardScaler()

    scaled = scaler.fit_transform(work)

    n_clusters = min(
        4,
        max(2, len(work) // 10)
    )

    n_clusters = min(
        n_clusters,
        len(work)
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        scaled
    )

    result = df.copy()

    result["Segment"] = labels + 1

    # Segment naming based on average values
    summary = result.groupby(
        "Segment"
    )[available].mean()

    names = {}

    monetary_col = "CLV_Monetary"

    if monetary_col in summary.columns:

        ranked = summary[
            monetary_col
        ].rank(
            ascending=False
        )

        for segment, rank in ranked.items():

            if rank <= 1:
                names[segment] = "Premium"
            elif rank <= 2:
                names[segment] = "High Value"
            elif rank >= len(ranked):
                names[segment] = "Low Value"
            else:
                names[segment] = "Growing"

    else:

        for segment in summary.index:
            names[segment] = f"Segment {segment}"

    result["Segment Name"] = result[
        "Segment"
    ].map(names)

    return result, summary


def currency(value):

    if pd.isna(value):
        return "₹0"

    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"

    if abs(value) >= 1000:
        return f"₹{value / 1000:.1f}K"

    return f"₹{value:,.0f}"


# ============================================================
# LOAD DATA
# ============================================================

dataset_path = find_dataset()

uploaded_file = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📊 CLV Studio")

    st.caption(
        "Customer Intelligence Platform"
    )

    st.divider()

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Dataset",
            "Customer Segments",
            "CLV Prediction",
            "Model Performance",
            "Business Insights"
        ]
    )

    st.divider()

    st.markdown("### Data Source")

    uploaded_file = st.file_uploader(
        "Upload customer dataset",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )

    st.divider()

    st.caption(
        "Machine Learning • Customer Analytics • CLV"
    )


# ============================================================
# DATA LOADING
# ============================================================

if uploaded_file is not None:

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    if extension == ".csv":
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    dataset_name = uploaded_file.name

elif dataset_path is not None:

    df = load_data(dataset_path)

    dataset_name = os.path.basename(
        dataset_path
    )

else:

    st.title("📊 CLV Studio")

    st.info(
        "Upload your customer dataset from the sidebar "
        "or place a CSV file inside the project folder."
    )

    st.markdown("""
    ### Expected data

    Your dataset should ideally contain columns such as:

    - Customer ID
    - Recency
    - Frequency
    - Monetary Value / Revenue
    - CLV

    The application automatically detects common column names.
    """)

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = clean_dataframe(df)

detected = detect_columns(df)

df_features = create_clv_features(
    df,
    detected
)

model_result = train_model(
    df_features,
    detected
)

segmented_df, segment_summary = create_segments(
    df_features
)


# ============================================================
# HEADER
# ============================================================

st.title("Customer Lifetime Value")

st.markdown(
    "### Machine Learning Customer Intelligence"
)

st.caption(
    f"Dataset: **{dataset_name}**  •  "
    f"{len(df):,} customer records  •  "
    f"{len(df.columns)} original features"
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Overview")

    total_customers = len(df)

    numeric = numeric_columns(
        df_features
    )

    if "CLV_Monetary" in df_features:

        total_revenue = df_features[
            "CLV_Monetary"
        ].sum()

        avg_value = df_features[
            "CLV_Monetary"
        ].mean()

    else:

        total_revenue = 0
        avg_value = 0

    if model_result is not None:

        predicted_clv = model_result[
            "model"
        ].predict(
            df_features[
                model_result["features"]
            ].fillna(0)
        )

        total_clv = predicted_clv.sum()

    else:

        total_clv = 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Customers",
            f"{total_customers:,}"
        )

    with c2:
        st.metric(
            "Total Value",
            currency(total_revenue)
        )

    with c3:
        st.metric(
            "Average Customer Value",
            currency(avg_value)
        )

    with c4:
        st.metric(
            "Predicted CLV",
            currency(total_clv)
        )

    st.write("")

    left, right = st.columns(
        [1.5, 1]
    )

    with left:

        st.subheader(
            "Customer Value Distribution"
        )

        if "CLV_Monetary" in df_features:

            fig, ax = plt.subplots(
                figsize=(9, 4)
            )

            ax.hist(
                df_features[
                    "CLV_Monetary"
                ].dropna(),
                bins=30
            )

            ax.set_xlabel(
                "Customer Value"
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

    with right:

        st.subheader(
            "ML System Status"
        )

        if model_result is not None:

            st.success(
                "Model trained successfully"
            )

            st.write(
                f"**Algorithm:** "
                f"Gradient Boosting Regressor"
            )

            st.write(
                f"**Features:** "
                f"{len(model_result['features'])}"
            )

            st.write(
                f"**R² Score:** "
                f"{model_result['r2']:.3f}"
            )

        else:

            st.warning(
                "Not enough suitable numeric data "
                "to train the model."
            )

    st.divider()

    st.subheader(
        "Detected Dataset Features"
    )

    detection_table = pd.DataFrame({
        "Feature": [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary Value",
            "CLV"
        ],
        "Detected Column": [
            detected["customer_id"],
            detected["recency"],
            detected["frequency"],
            detected["monetary"],
            detected["clv"]
        ]
    })

    st.dataframe(
        detection_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.subheader("Dataset Explorer")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Columns",
            len(df.columns)
        )

    with c3:
        st.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )

    st.write("")

    search = st.text_input(
        "Search columns",
        placeholder="Type a column name..."
    )

    display_df = df.copy()

    if search:

        matching_columns = [
            col
            for col in display_df.columns
            if search.lower()
            in str(col).lower()
        ]

        if matching_columns:

            display_df = display_df[
                matching_columns
            ]

        else:

            st.warning(
                "No matching columns found."
            )

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        height=500
    )

    st.download_button(
        "Download Dataset",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="clv_dataset.csv",
        mime="text/csv"
    )


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

elif page == "Customer Segments":

    st.subheader(
        "Customer Segmentation"
    )

    st.caption(
        "K-Means clustering groups customers according "
        "to their behavioural characteristics."
    )

    if segmented_df is None or "Segment" not in segmented_df:

        st.warning(
            "Not enough numeric customer features "
            "for segmentation."
        )

        st.stop()

    segments = segmented_df[
        "Segment Name"
    ].value_counts()

    cols = st.columns(
        min(4, len(segments))
    )

    for i, (name, count) in enumerate(
        segments.items()
    ):

        with cols[i % len(cols)]:

            st.metric(
                name,
                f"{count:,} customers"
            )

    st.write("")

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.subheader(
            "Segment Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        segments.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Customer Segment"
        )

        ax.set_ylabel(
            "Customers"
        )

        ax.tick_params(
            axis="x",
            rotation=25
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

    with right:

        st.subheader(
            "Segment Characteristics"
        )

        if segment_summary is not None:

            st.dataframe(
                segment_summary.round(2),
                use_container_width=True
            )

    st.divider()

    st.subheader(
        "Customer Segment Explorer"
    )

    selected_segment = st.selectbox(
        "Select segment",
        segmented_df[
            "Segment Name"
        ].dropna().unique()
    )

    segment_customers = segmented_df[
        segmented_df[
            "Segment Name"
        ] == selected_segment
    ]

    st.dataframe(
        segment_customers.head(100),
        use_container_width=True,
        height=400
    )


# ============================================================
# CLV PREDICTION
# ============================================================

elif page == "CLV Prediction":

    st.subheader(
        "Customer Lifetime Value Prediction"
    )

    if model_result is None:

        st.error(
            "The model could not be trained. "
            "Please provide sufficient numeric customer data."
        )

        st.stop()

    st.caption(
        "Enter customer behaviour to estimate future customer value."
    )

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.markdown(
            "#### Customer Behaviour"
        )

        recency = st.number_input(
            "Recency",
            min_value=0.0,
            value=30.0,
            step=1.0,
            help="Days since last purchase."
        )

        frequency = st.number_input(
            "Purchase Frequency",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        monetary = st.number_input(
            "Monetary Value",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        predict = st.button(
            "Predict Customer CLV",
            type="primary",
            use_container_width=True
        )

    with right:

        st.markdown(
            "#### Prediction"
        )

        if predict:

            input_data = pd.DataFrame({
                "CLV_Recency": [
                    recency
                ],
                "CLV_Frequency": [
                    frequency
                ],
                "CLV_Monetary": [
                    monetary
                ]
            })

            input_data = input_data[
                model_result["features"]
            ]

            prediction = model_result[
                "model"
            ].predict(
                input_data
            )[0]

            st.metric(
                "Estimated Customer Lifetime Value",
                currency(prediction)
            )

            if prediction > monetary:

                st.success(
                    "High potential customer. "
                    "Consider retention and loyalty strategies."
                )

            else:

                st.info(
                    "Focus on increasing engagement, "
                    "purchase frequency and retention."
                )

        else:

            st.info(
                "Enter customer information and "
                "click Predict Customer CLV."
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.subheader(
        "Model Performance"
    )

    if model_result is None:

        st.warning(
            "Model performance is unavailable."
        )

        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "R² Score",
            f"{model_result['r2']:.3f}"
        )

    with c2:

        st.metric(
            "MAE",
            f"{model_result['mae']:.2f}"
        )

    with c3:

        st.metric(
            "RMSE",
            f"{model_result['rmse']:.2f}"
        )

    st.write("")

    left, right = st.columns(
        [1.2, 1]
    )

    with left:

        st.subheader(
            "Actual vs Predicted"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            model_result["y_test"],
            model_result["predictions"],
            alpha=0.6
        )

        min_value = min(
            model_result["y_test"].min(),
            model_result["predictions"].min()
        )

        max_value = max(
            model_result["y_test"].max(),
            model_result["predictions"].max()
        )

        ax.plot(
            [min_value, max_value],
            [min_value, max_value]
        )

        ax.set_xlabel(
            "Actual CLV"
        )

        ax.set_ylabel(
            "Predicted CLV"
        )

        ax.grid(
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with right:

        st.subheader(
            "Model Information"
        )

        st.write(
            "**Algorithm**"
        )

        st.write(
            "Gradient Boosting Regressor"
        )

        st.write(
            "**Training samples**"
        )

        st.write(
            f"{len(model_result['X_train']):,}"
        )

        st.write(
            "**Testing samples**"
        )

        st.write(
            f"{len(model_result['X_test']):,}"
        )

        st.write(
            "**Features used**"
        )

        for feature in model_result[
            "features"
        ]:

            st.write(
                f"• {feature}"
            )

    st.divider()

    st.subheader(
        "Feature Importance"
    )

    importance = pd.DataFrame({
        "Feature": model_result[
            "features"
        ],
        "Importance": model_result[
            "model"
        ].feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.bar(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_ylabel(
        "Importance"
    )

    ax.set_xlabel(
        "Feature"
    )

    ax.tick_params(
        axis="x",
        rotation=20
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
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.subheader(
        "Business Intelligence"
    )

    st.caption(
        "Translate customer data into actionable insights."
    )

    if "CLV_Monetary" in segmented_df:

        avg_value = segmented_df[
            "CLV_Monetary"
        ].mean()

        high_value = segmented_df[
            segmented_df[
                "CLV_Monetary"
            ] >= avg_value
        ]

        low_value = segmented_df[
            segmented_df[
                "CLV_Monetary"
            ] < avg_value
        ]

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "### 💎 High-Value Customers"
            )

            st.metric(
                "Customers",
                f"{len(high_value):,}"
            )

            st.write(
                "Customers whose monetary value is "
                "above the dataset average."
            )

            st.success(
                "Recommended: loyalty programs, "
                "personalized offers and retention campaigns."
            )

        with c2:

            st.markdown(
                "### 📈 Growth Opportunity"
            )

            st.metric(
                "Customers",
                f"{len(low_value):,}"
            )

            st.write(
                "Customers currently below the average "
                "monetary value."
            )

            st.info(
                "Recommended: targeted campaigns, "
                "cross-selling and engagement strategies."
            )

    st.divider()

    st.subheader(
        "Recommended Actions"
    )

    actions = [
        (
            "Retain Premium Customers",
            "Identify high-value customers and prioritize retention."
        ),
        (
            "Increase Purchase Frequency",
            "Use personalized campaigns to encourage repeat purchases."
        ),
        (
            "Monitor Customer Recency",
            "Customers with increasing recency may require re-engagement."
        ),
        (
            "Segment Marketing Campaigns",
            "Use K-Means segments to create targeted customer campaigns."
        ),
        (
            "Use CLV for Decision Making",
            "Prioritize marketing resources based on predicted customer value."
        )
    ]

    for title, description in actions:

        with st.expander(title):

            st.write(description)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CLV Studio • Machine Learning Customer Intelligence • "
    "Built with Python, Streamlit & Scikit-learn"
)
