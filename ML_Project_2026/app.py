import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
# APPLICATION CONFIG
# ============================================================

DATA_FILE = "user_inputs.xlsx"


# ============================================================
# CUSTOM STREAMLIT THEME - NO HTML
# ============================================================

st.markdown(
    """
    # 📊 CLV Intelligence
    ### Customer Lifetime Value Analytics Platform
    """,
)

st.divider()


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
# MODEL FUNCTION
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
        ["Recency", "Frequency", "Monetary"]
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

    model.fit(X_train, y_train)

    return model, df


# ============================================================
# SAVE USER DATA
# ============================================================

def save_prediction(data):

    try:

        if os.path.exists(DATA_FILE):

            old_data = pd.read_excel(DATA_FILE)

            final_data = pd.concat(
                [old_data, data],
                ignore_index=True
            )

        else:

            final_data = data

        final_data.to_excel(
            DATA_FILE,
            index=False
        )

        return True

    except Exception as e:

        st.error(
            f"Unable to save data: {e}"
        )

        return False


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
            "👥 Customer Segmentation",
            "🔐 Admin Panel"
        ]
    )

    st.divider()

    st.success("🟢 System Online")

    st.caption(
        "Machine Learning\n"
        "Customer Analytics\n"
        "Secure Data Management"
    )


# ============================================================
# DASHBOARD
# ============================================================

if menu == "🏠 Dashboard":

    st.title("🏠 Dashboard")

    st.subheader(
        "Welcome to CLV Intelligence 👋"
    )

    st.write(
        """
        This application uses machine learning to estimate
        Customer Lifetime Value (CLV), analyze customer
        behaviour and divide customers into meaningful segments.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # KPI DATA
    # --------------------------------------------------------

    customer_count = 12
    average_clv = 1333
    highest_clv = 3000

    if os.path.exists(DATA_FILE):

        try:

            stored_data = pd.read_excel(
                DATA_FILE
            )

            if not stored_data.empty:

                customer_count = len(
                    stored_data
                )

                if "Predicted_CLV" in stored_data.columns:

                    average_clv = stored_data[
                        "Predicted_CLV"
                    ].mean()

                    highest_clv = stored_data[
                        "Predicted_CLV"
                    ].max()

        except:
            pass

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.subheader("📊 Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "👥 Customers Analyzed",
            customer_count
        )

    with c2:
        st.metric(
            "💰 Average CLV",
            f"${average_clv:,.0f}"
        )

    with c3:
        st.metric(
            "🏆 Highest CLV",
            f"${highest_clv:,.0f}"
        )

    with c4:
        st.metric(
            "🤖 ML Model",
            "Active"
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("🚀 Platform Capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info(
            """
            ### 🔮 CLV Prediction

            Predict customer lifetime value using:

            - Recency
            - Purchase Frequency
            - Monetary Value
            - Gradient Boosting
            """
        )

    with c2:

        st.info(
            """
            ### 👥 Customer Segmentation

            Discover customer groups using:

            - K-Means clustering
            - RFM analysis
            - Customer behaviour
            - Visual segmentation
            """
        )

    with c3:

        st.info(
            """
            ### 🔐 Admin Panel

            Administrators can:

            - View stored predictions
            - Inspect records
            - Download Excel data
            - Manage application data
            """
        )

    st.divider()

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader("⚙️ How the System Works")

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.metric(
            "01",
            "Input Data"
        )
        st.caption(
            "Enter customer behaviour."
        )

    with w2:
        st.metric(
            "02",
            "ML Processing"
        )
        st.caption(
            "Model analyzes customer data."
        )

    with w3:
        st.metric(
            "03",
            "CLV Prediction"
        )
        st.caption(
            "Estimate future customer value."
        )

    with w4:
        st.metric(
            "04",
            "Analytics"
        )
        st.caption(
            "Understand customer segments."
        )


# ============================================================
# CLV PREDICTOR
# ============================================================

elif menu == "🔮 CLV Predictor":

    st.title(
        "🔮 Customer Lifetime Value Predictor"
    )

    st.write(
        "Enter customer information to estimate their lifetime value."
    )

    st.divider()

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    with left:

        st.subheader(
            "📋 Customer Information"
        )

        recency = st.number_input(
            "📅 Recency (Days)",
            min_value=0,
            max_value=365,
            value=30,
            step=1
        )

        frequency = st.number_input(
            "🔄 Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )

        monetary = st.number_input(
            "💰 Monetary Value",
            min_value=0.0,
            max_value=100000.0,
            value=500.0,
            step=50.0
        )

        st.divider()

        predict_button = st.button(
            "🚀 Predict Customer CLV",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    with right:

        st.subheader(
            "👤 Customer Profile"
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

        st.info(
            """
            The model evaluates these three customer
            behaviour factors to estimate CLV.
            """
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        model, training_data = train_clv_model()

        user_data = pd.DataFrame(
            {
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            }
        )

        prediction = float(
            model.predict(user_data)[0]
        )

        user_data["Predicted_CLV"] = prediction

        saved = save_prediction(
            user_data
        )

        st.divider()

        st.subheader(
            "💎 Prediction Result"
        )

        if prediction >= 2000:

            value_level = "🟢 HIGH VALUE"

        elif prediction >= 1000:

            value_level = "🟡 MEDIUM VALUE"

        else:

            value_level = "🔵 STANDARD VALUE"

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Predicted Customer Lifetime Value",
                f"${prediction:,.2f}"
            )

        with c2:

            st.metric(
                "Customer Category",
                value_level
            )

        if saved:

            st.success(
                "✅ Prediction completed and stored successfully."
            )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        st.subheader(
            "📈 Customer Behaviour"
        )

        chart_data = pd.DataFrame(
            {
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
            }
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.bar(
            chart_data["Metric"],
            chart_data["Value"]
        )

        ax.set_title(
            "Customer Behaviour Profile"
        )

        ax.set_ylabel(
            "Value"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif menu == "👥 Customer Segmentation":

    st.title(
        "👥 Customer Segmentation"
    )

    st.write(
        """
        Upload a customer dataset and use K-Means clustering
        to identify customer groups.
        """
    )

    st.divider()

    st.subheader(
        "📂 Upload Customer Dataset"
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "📌 Upload a CSV file to start segmentation."
        )

    else:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"✅ Dataset loaded successfully — {len(df)} records found."
            )

            st.subheader(
                "👀 Dataset Preview"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.divider()

            st.subheader(
                "⚙️ Segmentation Configuration"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                r_column = st.selectbox(
                    "📅 Recency Column",
                    df.columns
                )

            with c2:

                f_column = st.selectbox(
                    "🔄 Frequency Column",
                    df.columns
                )

            with c3:

                m_column = st.selectbox(
                    "💰 Monetary Column",
                    df.columns
                )

            clusters = st.slider(
                "🎯 Number of Clusters",
                min_value=2,
                max_value=6,
                value=3
            )

            st.divider()

            run_segmentation = st.button(
                "🚀 Run Segmentation",
                type="primary",
                use_container_width=True
            )

            if run_segmentation:

                X = df[
                    [
                        r_column,
                        f_column,
                        m_column
                    ]
                ].copy()

                X = X.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                valid_rows = X.notna().all(
                    axis=1
                )

                X = X.loc[
                    valid_rows
                ]

                result_df = df.loc[
                    valid_rows
                ].copy()

                if len(X) < clusters:

                    st.error(
                        "❌ Number of valid records is smaller than the selected clusters."
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

                    result_df["Cluster"] = kmeans.fit_predict(
                        X_scaled
                    )

                    st.success(
                        "✅ Customer segmentation completed."
                    )

                    # ------------------------------------------------
                    # KPI
                    # ------------------------------------------------

                    st.subheader(
                        "📊 Segmentation Overview"
                    )

                    a, b, c, d = st.columns(4)

                    with a:
                        st.metric(
                            "👥 Customers",
                            len(result_df)
                        )

                    with b:
                        st.metric(
                            "🎯 Clusters",
                            clusters
                        )

                    with c:
                        st.metric(
                            "💰 Avg Monetary",
                            f"${X[m_column].mean():,.2f}"
                        )

                    with d:
                        st.metric(
                            "📅 Avg Recency",
                            f"{X[r_column].mean():,.1f}"
                        )

                    # ------------------------------------------------
                    # GRAPH
                    # ------------------------------------------------

                    st.subheader(
                        "📈 Customer Segmentation Map"
                    )

                    fig, ax = plt.subplots(
                        figsize=(11, 5)
                    )

                    scatter = ax.scatter(
                        result_df[r_column],
                        result_df[m_column],
                        c=result_df["Cluster"],
                        cmap="viridis",
                        s=80,
                        alpha=0.8
                    )

                    ax.set_xlabel(
                        r_column
                    )

                    ax.set_ylabel(
                        m_column
                    )

                    ax.set_title(
                        "Customer Segmentation"
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

                    # ------------------------------------------------
                    # SUMMARY
                    # ------------------------------------------------

                    st.subheader(
                        "📋 Segment Summary"
                    )

                    summary = result_df.groupby(
                        "Cluster"
                    )[
                        [
                            r_column,
                            f_column,
                            m_column
                        ]
                    ].mean().round(2)

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # FULL DATA
                    # ------------------------------------------------

                    st.subheader(
                        "👥 Classified Customers"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # DOWNLOAD
                    # ------------------------------------------------

                    csv_output = result_df.to_csv(
                        index=False
                    )

                    st.download_button(
                        "📥 Download Segmented Dataset",
                        csv_output,
                        "segmented_customers.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as error:

            st.error(
                f"❌ Error processing dataset: {error}"
            )


# ============================================================
# ADMIN PANEL
# ============================================================

elif menu == "🔐 Admin Panel":

    st.title(
        "🔐 Admin Panel"
    )

    st.write(
        "Secure area for viewing stored CLV prediction data."
    )

    st.divider()

    # ========================================================
    # BLOCKED
    # ========================================================

    if st.session_state.blocked:

        st.error(
            "🚫 Access blocked after 3 incorrect password attempts."
        )

        st.warning(
            "Please contact the administrator for access."
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
            "🟢 Admin access granted."
        )

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.admin_logged_in = False

            st.rerun()

        st.divider()

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        st.subheader(
            "📊 Stored Customer Predictions"
        )

        if os.path.exists(DATA_FILE):

            try:

                admin_data = pd.read_excel(
                    DATA_FILE
                )

                if admin_data.empty:

                    st.info(
                        "No customer data available."
                    )

                else:

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "👥 Total Records",
                            len(admin_data)
                        )

                    with c2:

                        if "Predicted_CLV" in admin_data.columns:

                            st.metric(
                                "💰 Average CLV",
                                f"${admin_data['Predicted_CLV'].mean():,.2f}"
                            )

                    with c3:

                        if "Predicted_CLV" in admin_data.columns:

                            st.metric(
                                "🏆 Highest CLV",
                                f"${admin_data['Predicted_CLV'].max():,.2f}"
                            )

                    st.dataframe(
                        admin_data,
                        use_container_width=True
                    )

                    st.divider()

                    with open(
                        DATA_FILE,
                        "rb"
                    ) as excel_file:

                        st.download_button(
                            "📥 Download Excel Data",
                            excel_file,
                            file_name="user_inputs.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            except Exception as error:

                st.error(
                    f"Unable to read Excel file: {error}"
                )

        else:

            st.info(
                "📭 No prediction data has been generated yet."
            )

        # ----------------------------------------------------
        # FILES
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📂 Application Files"
        )

        files = os.listdir()

        for file_name in files:

            st.write(
                f"📄 {file_name}"
            )

    # ========================================================
    # LOGIN
    # ========================================================

    else:

        st.subheader(
            "🔑 Administrator Login"
        )

        st.info(
            "Authorized users only."
        )

        password = st.text_input(
            "Enter Administrator Password",
            type="password"
        )

        login_button = st.button(
            "🔐 Login",
            type="primary",
            use_container_width=True
        )

        if login_button:

            if password.strip() == "admin123":

                st.session_state.admin_logged_in = True

                st.session_state.attempts = 0

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
                        f"❌ Wrong password. "
                        f"Attempts remaining: {remaining}"
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

st.divider()

st.caption(
    "📊 CLV Intelligence • Python • Streamlit • Pandas • Scikit-Learn"
)
