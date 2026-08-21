import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CLV Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "user_inputs.xlsx"


# =========================================================
# SESSION STATE
# =========================================================

for key, value in {
    "admin_logged_in": False,
    "attempts": 0,
    "blocked": False,
    "last_prediction": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def train_model():

    df = pd.DataFrame({
        "Recency": [10,20,5,30,15,40,25,8,60,12,35,18],
        "Frequency": [5,3,10,2,7,1,4,12,2,8,3,6],
        "Monetary": [500,300,1000,200,700,100,400,1500,250,900,350,650],
        "CLV": [1200,700,2500,400,1600,200,900,3000,500,2000,800,1400]
    })

    X = df[["Recency","Frequency","Monetary"]]
    y = df["CLV"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    return model


# =========================================================
# SAVE DATA
# =========================================================

def save_data(data):

    if os.path.exists(DATA_FILE):
        old = pd.read_excel(DATA_FILE)
        data = pd.concat([old, data], ignore_index=True)

    data.to_excel(DATA_FILE, index=False)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📊 CLV Intelligence")
    st.caption("Customer Analytics Platform")

    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 CLV Predictor",
            "👥 Segmentation",
            "🔐 Admin"
        ]
    )

    st.divider()

    st.success("🟢 System Online")

    st.caption("Machine Learning")
    st.caption("Customer Analytics")
    st.caption("Secure Data Management")


# =========================================================
# DASHBOARD
# =========================================================

if menu == "🏠 Dashboard":

    st.title("Customer Lifetime Value Dashboard")

    st.write(
        "Analyze customers, predict lifetime value and identify customer segments using machine learning."
    )

    st.divider()

    total = 0
    avg = 0
    high = 0

    if os.path.exists(DATA_FILE):
        try:
            data = pd.read_excel(DATA_FILE)
            total = len(data)

            if "Predicted_CLV" in data.columns:
                avg = data["Predicted_CLV"].mean()
                high = data["Predicted_CLV"].max()
        except:
            pass

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👥 Customers", total)
    c2.metric("💰 Average CLV", f"${avg:,.0f}")
    c3.metric("🏆 Highest CLV", f"${high:,.0f}")
    c4.metric("🤖 ML Model", "Active")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("🚀 Quick Actions")

        st.info("🔮 Predict Customer Lifetime Value")
        st.info("👥 Segment Customers with K-Means")
        st.info("🔐 View Stored Predictions")

    with right:

        st.subheader("📈 CLV Overview")

        if os.path.exists(DATA_FILE):

            try:

                data = pd.read_excel(DATA_FILE)

                if "Predicted_CLV" in data.columns and len(data):

                    fig, ax = plt.subplots()

                    ax.plot(
                        data["Predicted_CLV"].values,
                        marker="o"
                    )

                    ax.set_title("Predicted CLV Trend")
                    ax.set_xlabel("Prediction")
                    ax.set_ylabel("CLV")
                    ax.grid(alpha=0.3)

                    st.pyplot(fig)

                else:
                    st.info("No prediction data yet.")

            except:
                st.info("No prediction data yet.")

        else:
            st.info("No prediction data yet.")

    st.divider()

    st.subheader("How It Works")

    a, b, c = st.columns(3)

    a.info("### 1️⃣ Input\nRecency, Frequency and Monetary value.")
    b.info("### 2️⃣ ML Model\nGradient Boosting predicts CLV.")
    c.info("### 3️⃣ Analytics\nView value and customer segments.")


# =========================================================
# CLV PREDICTOR
# =========================================================

elif menu == "🔮 CLV Predictor":

    st.title("🔮 Customer Lifetime Value Predictor")

    st.caption("Enter customer purchase behaviour to estimate future value.")

    st.divider()

    left, right = st.columns([1,1])

    with left:

        st.subheader("Customer Inputs")

        recency = st.number_input("📅 Recency (Days)", 0,365,30)
        frequency = st.number_input("🔄 Purchase Frequency",1,100,5)
        monetary = st.number_input("💰 Monetary Value",0.0,100000.0,500.0,step=50.0)

        predict = st.button(
            "🚀 Predict CLV",
            type="primary",
            use_container_width=True
        )

    with right:

        st.subheader("Customer Profile")

        st.metric("Recency",f"{recency} Days")
        st.metric("Frequency",frequency)
        st.metric("Monetary",f"${monetary:,.2f}")

        st.info(
            "The model evaluates RFM behaviour to estimate the customer's lifetime value."
        )

    if predict:

        model = train_model()

        user = pd.DataFrame({
            "Recency":[recency],
            "Frequency":[frequency],
            "Monetary":[monetary]
        })

        prediction = float(model.predict(user)[0])

        user["Predicted_CLV"] = prediction

        save_data(user)

        st.session_state.last_prediction = prediction

    if st.session_state.last_prediction is not None:

        prediction = st.session_state.last_prediction

        st.divider()

        st.subheader("Prediction Result")

        r1, r2 = st.columns(2)

        r1.metric(
            "💰 Predicted CLV",
            f"${prediction:,.2f}"
        )

        category = (
            "High Value" if prediction >= 2000
            else "Medium Value" if prediction >= 1000
            else "Standard Value"
        )

        r2.metric("Customer Category", category)

        st.success("Prediction saved successfully.")

        fig, ax = plt.subplots()

        ax.bar(
            ["Recency","Frequency","Monetary"],
            [recency,frequency,monetary]
        )

        ax.set_title("Customer Behaviour Profile")

        st.pyplot(fig)


# =========================================================
# SEGMENTATION
# =========================================================

elif menu == "👥 Segmentation":

    st.title("👥 Customer Segmentation")

    st.caption("Upload a CSV dataset and group customers using K-Means clustering.")

    st.divider()

    file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if file:

        df = pd.read_csv(file)

        st.success(f"Loaded {len(df)} customer records.")

        st.dataframe(df.head(),use_container_width=True)

        st.divider()

        c1,c2,c3 = st.columns(3)

        r = c1.selectbox("Recency Column",df.columns)
        f = c2.selectbox("Frequency Column",df.columns)
        m = c3.selectbox("Monetary Column",df.columns)

        clusters = st.slider(
            "Number of Segments",
            2,6,3
        )

        if st.button(
            "Run Segmentation",
            type="primary",
            use_container_width=True
        ):

            X = df[[r,f,m]].apply(
                pd.to_numeric,
                errors="coerce"
            ).dropna()

            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(
                n_clusters=clusters,
                random_state=42,
                n_init=10
            )

            df = df.loc[X.index].copy()

            df["Cluster"] = kmeans.fit_predict(X_scaled)

            st.success("Segmentation completed.")

            a,b,c,d = st.columns(4)

            a.metric("Customers",len(df))
            b.metric("Segments",clusters)
            c.metric("Average Monetary",f"${X[m].mean():,.0f}")
            d.metric("Average Recency",f"{X[r].mean():.1f}")

            fig,ax = plt.subplots()

            scatter = ax.scatter(
                df[r],
                df[m],
                c=df["Cluster"],
                cmap="viridis"
            )

            ax.set_xlabel(r)
            ax.set_ylabel(m)
            ax.set_title("Customer Segmentation Map")

            plt.colorbar(scatter,ax=ax)

            st.pyplot(fig)

            st.subheader("Segment Summary")

            summary = df.groupby("Cluster")[[r,f,m]].mean().round(2)

            st.dataframe(summary,use_container_width=True)

            st.subheader("Classified Customers")

            st.dataframe(df,use_container_width=True)

            st.download_button(
                "Download Segmented CSV",
                df.to_csv(index=False),
                "segmented_customers.csv",
                "text/csv"
            )


# =========================================================
# ADMIN
# =========================================================

elif menu == "🔐 Admin":

    st.title("🔐 Admin Panel")

    st.divider()

    if st.session_state.blocked:

        st.error("Access blocked after 3 incorrect attempts.")

        st.link_button(
            "Contact Administrator",
            "https://mail.google.com/mail/?view=cm&fs=1&to=atharavshende999@gmail.com"
        )

        st.stop()

    if not st.session_state.admin_logged_in:

        password = st.text_input(
            "Administrator Password",
            type="password"
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True
        ):

            if password == "admin123":

                st.session_state.admin_logged_in = True
                st.session_state.attempts = 0

                st.rerun()

            else:

                st.session_state.attempts += 1

                remaining = 3 - st.session_state.attempts

                if remaining <= 0:

                    st.session_state.blocked = True

                    st.rerun()

                st.error(
                    f"Wrong password. {remaining} attempts remaining."
                )

    else:

        st.success("Admin access granted.")

        if st.button("Logout"):

            st.session_state.admin_logged_in = False

            st.rerun()

        if os.path.exists(DATA_FILE):

            data = pd.read_excel(DATA_FILE)

            c1,c2,c3 = st.columns(3)

            c1.metric("Records",len(data))

            if "Predicted_CLV" in data.columns:

                c2.metric(
                    "Average CLV",
                    f"${data['Predicted_CLV'].mean():,.2f}"
                )

                c3.metric(
                    "Highest CLV",
                    f"${data['Predicted_CLV'].max():,.2f}"
                )

            st.dataframe(data,use_container_width=True)

            with open(DATA_FILE,"rb") as f:

                st.download_button(
                    "Download Excel",
                    f,
                    "user_inputs.xlsx"
                )

        else:

            st.info("No prediction records found.")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CLV Intelligence Dashboard • Python • Streamlit • Scikit-Learn"
)
