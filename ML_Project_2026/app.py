import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value (CLV) App")


# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Select Mode",
    [
        "🏠 Home",
        "🔮 CLV Predictor",
        "📊 Segmentation Dashboard",
        "🔐 Admin Panel"
    ]
)


# ---------------- HOME ----------------
if menu == "🏠 Home":

    st.header("Project Overview")
    st.write("""
    ✔ Predict CLV using ML  
    ✔ Store user data (hidden)  
    ✔ Customer segmentation  
    ✔ Admin-only access  
    """)


# ---------------- CLV PREDICTOR ----------------
elif menu == "🔮 CLV Predictor":

    st.header("🔮 Predict Customer Lifetime Value")

    recency = st.number_input("Recency (Days)", 0, 365, 30)
    frequency = st.number_input("Frequency", 1, 100, 5)
    monetary = st.number_input("Monetary Value", 0.0, 10000.0, 500.0)

    if st.button("Predict CLV"):

        # Sample dataset
        data = {
            "Recency":[10,20,5,30,15,40,25,8,60,12,35,18],
            "Frequency":[5,3,10,2,7,1,4,12,2,8,3,6],
            "Monetary":[500,300,1000,200,700,100,400,1500,250,900,350,650],
            "CLV":[1200,700,2500,400,1600,200,900,3000,500,2000,800,1400]
        }

        df = pd.DataFrame(data)

        X = df[["Recency","Frequency","Monetary"]]
        y = df["CLV"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        user_data = pd.DataFrame({
            "Recency":[recency],
            "Frequency":[frequency],
            "Monetary":[monetary]
        })

        prediction = model.predict(user_data)
        user_data["Predicted_CLV"] = prediction[0]

        # -------- SAVE TO EXCEL --------
        file_name = "user_inputs.xlsx"

        try:
            old = pd.read_excel(file_name)
            new = pd.concat([old, user_data], ignore_index=True)
            new.to_excel(file_name, index=False)
        except:
            user_data.to_excel(file_name, index=False)

        st.success(f"💰 Predicted CLV: ${prediction[0]:.2f}")


# ---------------- SEGMENTATION ----------------
elif menu == "📊 Segmentation Dashboard":

    st.header("📊 Customer Segmentation")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        r = st.selectbox("Recency column", df.columns)
        f = st.selectbox("Frequency column", df.columns)
        m = st.selectbox("Monetary column", df.columns)

        if st.button("Run Segmentation"):

            X = df[[r,f,m]]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=3, random_state=42)
            df["Cluster"] = kmeans.fit_predict(X_scaled)

            st.dataframe(df.head())

            fig, ax = plt.subplots()
            ax.scatter(df[r], df[m], c=df["Cluster"])
            ax.set_xlabel("Recency")
            ax.set_ylabel("Monetary")
            ax.set_title("Customer Segmentation")

            st.pyplot(fig)


# ---------------- ADMIN PANEL ----------------
elif menu == "🔐 Admin Panel":

    st.header("🔐 Admin Access")

    # -------- SESSION STATE --------
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0

    if "blocked" not in st.session_state:
        st.session_state.blocked = False

    # -------- BLOCKED USER --------
    if st.session_state.blocked:
        st.error("🚫 Too many wrong attempts. Access blocked.")

        email = "atharavshende999@gmail.com"
        subject = "Access Request for CLV App"

        st.markdown("### 📩 Contact Admin")
        st.markdown(
            f'<a href="mailto:{email}?subject={subject}">'
            f'<button>📧 Contact Admin</button></a>',
            unsafe_allow_html=True
        )

        st.stop()

    # -------- LOGIN --------
    password = st.text_input("Enter Password", type="password")

    if st.button("Login", use_container_width=True):

        if password.strip() == "admin123":
            st.session_state.attempts = 0
            st.success("✅ Access Granted")

            # -------- ADMIN CONTENT --------
            st.subheader("📂 Server Files")
            st.write(os.listdir())

            try:
                df = pd.read_excel("user_inputs.xlsx")
                st.subheader("📄 Stored User Data")
                st.dataframe(df)
            except:
                st.warning("No data found")

            try:
                with open("user_inputs.xlsx", "rb") as f:
                    st.download_button(
                        "📥 Download Excel",
                        f,
                        file_name="user_inputs.xlsx"
                    )
            except:
                pass

        else:
            st.session_state.attempts += 1

            remaining = 3 - st.session_state.attempts

            if remaining > 0:
                st.error(f"❌ Wrong password! Attempts left: {remaining}")
            else:
                st.session_state.blocked = True
                st.error("🚫 You are blocked after 3 wrong attempts")
                st.rerun()
