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
    page_title="CLV AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "user_inputs.xlsx"


# ============================================================
# SESSION STATE
# ============================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "attempts" not in st.session_state:
    st.session_state.attempts = 0

if "blocked" not in st.session_state:
    st.session_state.blocked = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
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

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(data):

    try:

        if os.path.exists(DATA_FILE):

            old_data = pd.read_excel(
                DATA_FILE
            )

            final_data = pd.concat(
                [
                    old_data,
                    data
                ],
                ignore_index=True
            )

        else:

            final_data = data

        final_data.to_excel(
            DATA_FILE,
            index=False
        )

        return True

    except Exception as error:

        st.error(
            f"Unable to save prediction: {error}"
        )

        return False


# ============================================================
# LOAD STORED DATA
# ============================================================

def load_prediction_data():

    if not os.path.exists(DATA_FILE):
        return None

    try:

        df = pd.read_excel(
            DATA_FILE
        )

        return df

    except:

        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 CLV AI")

    st.caption(
        "Customer Lifetime Value"
    )

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Home",
            "🔮 CLV Predictor",
            "👥 Customer Segmentation",
            "📈 Analytics",
            "🔐 Admin Panel"
        ]
    )

    st.divider()

    st.success(
        "🟢 System Online"
    )

    st.caption(
        "AI Powered Analytics"
    )

    st.caption(
        "Python + Streamlit"
    )

    st.caption(
        "Machine Learning"
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "Customer Lifetime Value"
    )

    st.subheader(
        "AI-Powered Customer Analytics Platform"
    )

    st.write(
        """
        Understand your customers, predict their future value,
        and discover valuable customer segments using machine
        learning.
        """
    )

    st.divider()

    # ========================================================
    # HERO VISUAL
    # ========================================================

    st.subheader(
        "🚀 Intelligent Customer Analytics"
    )

    hero_col1, hero_col2 = st.columns(
        [1.5, 1]
    )

    with hero_col1:

        st.info(
            """
            ### 🧠 AI Customer Intelligence

            Our system analyzes:

            **Recency** → When did the customer last purchase?

            **Frequency** → How often does the customer purchase?

            **Monetary** → How much does the customer spend?

            These factors are used to estimate Customer
            Lifetime Value.
            """
        )

        st.button(
            "🔮 Start CLV Prediction",
            use_container_width=True
        )

    with hero_col2:

        st.image(
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=80",
            caption="Customer Analytics Dashboard",
            use_container_width=True
        )

    st.divider()

    # ========================================================
    # KPI
    # ========================================================

    stored = load_prediction_data()

    total_customers = 0
    average_clv = 0
    maximum_clv = 0

    if stored is not None and not stored.empty:

        total_customers = len(stored)

        if "Predicted_CLV" in stored.columns:

            average_clv = stored[
                "Predicted_CLV"
            ].mean()

            maximum_clv = stored[
                "Predicted_CLV"
            ].max()

    st.subheader(
        "📊 Business Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Customers",
        total_customers
    )

    c2.metric(
        "💰 Average CLV",
        f"${average_clv:,.0f}"
    )

    c3.metric(
        "🏆 Highest CLV",
        f"${maximum_clv:,.0f}"
    )

    c4.metric(
        "🤖 ML Model",
        "Active"
    )

    st.divider()

    # ========================================================
    # FEATURES WITH IMAGES
    # ========================================================

    st.subheader(
        "✨ Platform Features"
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.image(
            "https://images.unsplash.com/photo-1556761175-b413da4baf72?auto=format&fit=crop&w=800&q=80",
            use_container_width=True
        )

        st.subheader(
            "🔮 CLV Prediction"
        )

        st.write(
            "Predict the future value of individual customers using machine learning."
        )

    with f2:

        st.image(
            "https://images.unsplash.com/photo-1521737711867-e3b97375f902?auto=format&fit=crop&w=800&q=80",
            use_container_width=True
        )

        st.subheader(
            "👥 Customer Segmentation"
        )

        st.write(
            "Group customers according to their purchasing behaviour."
        )

    with f3:

        st.image(
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=800&q=80",
            use_container_width=True
        )

        st.subheader(
            "📈 Business Analytics"
        )

        st.write(
            "Analyze customer value and make better business decisions."
        )

    st.divider()

    st.subheader(
        "⚙️ How It Works"
    )

    a, b, c, d = st.columns(4)

    with a:

        st.info(
            """
            ### 01

            📥 **Input**

            Customer RFM data
            """

        )

    with b:

        st.info(
            """
            ### 02

            🧠 **Process**

            Machine Learning
            """

        )

    with c:

        st.info(
            """
            ### 03

            💰 **Predict**

            Customer Lifetime Value
            """

        )

    with d:

        st.info(
            """
            ### 04

            📊 **Analyze**

            Customer segments
            """
        )


# ============================================================
# CLV PREDICTOR
# ============================================================

elif page == "🔮 CLV Predictor":

    st.title(
        "🔮 CLV Predictor"
    )

    st.write(
        "Estimate the future lifetime value of a customer."
    )

    st.divider()

    left, right = st.columns(
        [1, 1]
    )

    # ========================================================
    # INPUT
    # ========================================================

    with left:

        st.subheader(
            "📋 Customer Information"
        )

        recency = st.number_input(
            "📅 Recency",
            min_value=0,
            max_value=365,
            value=30
        )

        st.caption(
            "Days since the last purchase."
        )

        frequency = st.number_input(
            "🔄 Purchase Frequency",
            min_value=1,
            max_value=100,
            value=5
        )

        st.caption(
            "Number of purchases."
        )

        monetary = st.number_input(
            "💰 Monetary Value",
            min_value=0.0,
            max_value=100000.0,
            value=500.0,
            step=50.0
        )

        st.caption(
            "Total customer spending."
        )

        predict = st.button(
            "🚀 Predict Customer CLV",
            type="primary",
            use_container_width=True
        )

    # ========================================================
    # PROFILE
    # ========================================================

    with right:

        st.subheader(
            "👤 Customer Profile"
        )

        st.metric(
            "Recency",
            f"{recency} Days"
        )

        st.metric(
            "Frequency",
            frequency
        )

        st.metric(
            "Monetary",
            f"${monetary:,.2f}"
        )

        st.image(
            "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?auto=format&fit=crop&w=900&q=80",
            caption="Customer Purchase Behaviour",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict:

        model = train_clv_model()

        user_data = pd.DataFrame(
            {
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            }
        )

        prediction = float(
            model.predict(
                user_data
            )[0]
        )

        user_data[
            "Predicted_CLV"
        ] = prediction

        save_prediction(
            user_data
        )

        st.session_state.last_prediction = prediction

    # ========================================================
    # RESULT
    # ========================================================

    if st.session_state.last_prediction is not None:

        prediction = (
            st.session_state.last_prediction
        )

        st.divider()

        st.subheader(
            "💎 Prediction Result"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "💰 Predicted CLV",
            f"${prediction:,.2f}"
        )

        if prediction >= 2000:

            category = "🟢 High Value"

        elif prediction >= 1000:

            category = "🟡 Medium Value"

        else:

            category = "🔵 Standard Value"

        r2.metric(
            "Customer Category",
            category
        )

        r3.metric(
            "Model",
            "Gradient Boosting"
        )

        st.success(
            "✅ Prediction completed and saved."
        )

        st.subheader(
            "📊 Customer Behaviour"
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
            "Customer RFM Profile"
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

elif page == "👥 Customer Segmentation":

    st.title(
        "👥 Customer Segmentation"
    )

    st.write(
        "Discover customer groups using K-Means clustering."
    )

    st.divider()

    st.image(
        "https://images.unsplash.com/photo-1552664730-d307ca884978?auto=format&fit=crop&w=1400&q=80",
        caption="Customer Segmentation & Team Analytics",
        use_container_width=True
    )

    st.subheader(
        "📂 Upload Customer Dataset"
    )

    file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if file:

        try:

            df = pd.read_csv(
                file
            )

            st.success(
                f"Dataset loaded: {len(df)} records"
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
                "⚙️ Configure Segmentation"
            )

            c1, c2, c3 = st.columns(3)

            r = c1.selectbox(
                "📅 Recency Column",
                df.columns
            )

            f = c2.selectbox(
                "🔄 Frequency Column",
                df.columns
            )

            m = c3.selectbox(
                "💰 Monetary Column",
                df.columns
            )

            clusters = st.slider(
                "🎯 Number of Customer Segments",
                2,
                6,
                3
            )

            run = st.button(
                "🚀 Run Segmentation",
                type="primary",
                use_container_width=True
            )

            if run:

                X = df[
                    [r, f, m]
                ].copy()

                X = X.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                valid = X.notna().all(
                    axis=1
                )

                X = X.loc[
                    valid
                ]

                result = df.loc[
                    valid
                ].copy()

                if len(X) < clusters:

                    st.error(
                        "Not enough valid records."
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

                    result[
                        "Cluster"
                    ] = kmeans.fit_predict(
                        X_scaled
                    )

                    st.success(
                        "✅ Segmentation completed successfully."
                    )

                    st.divider()

                    st.subheader(
                        "📊 Segment Overview"
                    )

                    a, b, c, d = st.columns(4)

                    a.metric(
                        "👥 Customers",
                        len(result)
                    )

                    b.metric(
                        "🎯 Segments",
                        clusters
                    )

                    c.metric(
                        "💰 Avg Spending",
                        f"${X[m].mean():,.0f}"
                    )

                    d.metric(
                        "📅 Avg Recency",
                        f"{X[r].mean():.1f}"
                    )

                    st.subheader(
                        "📈 Customer Segmentation Map"
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

                    ax.set_xlabel(
                        r
                    )

                    ax.set_ylabel(
                        m
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

                    st.subheader(
                        "📋 Segment Summary"
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

                    st.subheader(
                        "👥 Classified Customers"
                    )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Download Segmented Dataset",
                        result.to_csv(
                            index=False
                        ),
                        "segmented_customers.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as error:

            st.error(
                f"Error: {error}"
            )

    else:

        st.info(
            "Upload a CSV dataset above to begin."
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.title(
        "📈 Customer Analytics"
    )

    st.write(
        "Analyze stored CLV prediction results."
    )

    st.divider()

    data = load_prediction_data()

    if data is None or data.empty:

        st.info(
            "📭 No prediction data available yet."
        )

        st.image(
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1400&q=80",
            caption="Analytics will appear here after predictions are generated.",
            use_container_width=True
        )

    else:

        if "Predicted_CLV" in data.columns:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "👥 Customers",
                len(data)
            )

            c2.metric(
                "💰 Average CLV",
                f"${data['Predicted_CLV'].mean():,.2f}"
            )

            c3.metric(
                "🏆 Maximum CLV",
                f"${data['Predicted_CLV'].max():,.2f}"
            )

            c4.metric(
                "📉 Minimum CLV",
                f"${data['Predicted_CLV'].min():,.2f}"
            )

            st.divider()

            st.subheader(
                "📊 CLV Distribution"
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.hist(
                data["Predicted_CLV"],
                bins=10
            )

            ax.set_title(
                "Customer Lifetime Value Distribution"
            )

            ax.set_xlabel(
                "Predicted CLV"
            )

            ax.set_ylabel(
                "Number of Customers"
            )

            ax.grid(
                alpha=0.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            st.subheader(
                "📈 Prediction Trend"
            )

            fig2, ax2 = plt.subplots(
                figsize=(10, 5)
            )

            ax2.plot(
                data["Predicted_CLV"],
                marker="o"
            )

            ax2.set_title(
                "Customer CLV Trend"
            )

            ax2.set_xlabel(
                "Customer"
            )

            ax2.set_ylabel(
                "Predicted CLV"
            )

            ax2.grid(
                alpha=0.2
            )

            st.pyplot(
                fig2,
                use_container_width=True
            )

            st.subheader(
                "📋 Prediction Records"
            )

            st.dataframe(
                data,
                use_container_width=True
            )

        else:

            st.warning(
                "Predicted CLV column not found."
            )


# ============================================================
# ADMIN PANEL
# ============================================================

elif page == "🔐 Admin Panel":

    st.title(
        "🔐 Admin Panel"
    )

    st.write(
        "Secure management area for customer prediction data."
    )

    st.divider()

    # ========================================================
    # BLOCKED
    # ========================================================

    if st.session_state.blocked:

        st.error(
            "🚫 Access blocked after 3 incorrect attempts."
        )

        st.warning(
            "Please contact the administrator."
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

    if not st.session_state.admin_logged_in:

        st.image(
            "https://images.unsplash.com/photo-1563013544-824ae1b704d3?auto=format&fit=crop&w=900&q=80",
            caption="Secure Administration",
            use_container_width=True
        )

        st.subheader(
            "🔑 Administrator Login"
        )

        password = st.text_input(
            "Enter Password",
            type="password"
        )

        login = st.button(
            "🔐 Secure Login",
            type="primary",
            use_container_width=True
        )

        if login:

            if password == "admin123":

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
                        f"❌ Incorrect password. "
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
            "🟢 Administrator authenticated."
        )

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.admin_logged_in = False

            st.rerun()

        st.divider()

        data = load_prediction_data()

        if data is not None and not data.empty:

            st.subheader(
                "📊 Stored Customer Data"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "👥 Records",
                len(data)
            )

            if "Predicted_CLV" in data.columns:

                c2.metric(
                    "💰 Average CLV",
                    f"${data['Predicted_CLV'].mean():,.2f}"
                )

                c3.metric(
                    "🏆 Highest CLV",
                    f"${data['Predicted_CLV'].max():,.2f}"
                )

            st.dataframe(
                data,
                use_container_width=True
            )

            st.divider()

            with open(
                DATA_FILE,
                "rb"
            ) as excel_file:

                st.download_button(
                    "📥 Download Excel",
                    excel_file,
                    file_name="user_inputs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        else:

            st.info(
                "No customer prediction records found."
            )

        st.divider()

        st.subheader(
            "📂 Application Files"
        )

        for filename in os.listdir():

            st.write(
                f"📄 {filename}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📊 CLV AI Analytics • Customer Lifetime Value • "
    "Python • Streamlit • Scikit-Learn"
)
