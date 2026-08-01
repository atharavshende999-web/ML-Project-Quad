import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ---------------- CONFIG ----------------
st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value (CLV) App")

PASSWORD_FILE = "password.json"


# ---------------- PASSWORD STORAGE ----------------
def load_password():
    if os.path.exists(PASSWORD_FILE):
        with open(PASSWORD_FILE, "r") as f:
            return json.load(f)["password"]
    return "admin123"

def save_password(new_pass):
    with open(PASSWORD_FILE, "w") as f:
        json.dump({"password": new_pass}, f)


# ---------------- OTP FUNCTION ----------------
def generate_otp():
    return random.randint(100000, 999999)


# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Select Mode",
    ["🏠 Home", "🔮 CLV Predictor", "📊 Segmentation", "🔐 Admin Panel"]
)


# ---------------- HOME ----------------
if menu == "🏠 Home":
    st.header("Project Overview")
    st.write("""
    ✔ CLV Prediction using ML  
    ✔ Store user data  
    ✔ Customer segmentation  
    ✔ OTP-based password reset  
    """)


# ---------------- CLV PREDICTOR ----------------
elif menu == "🔮 CLV Predictor":

    st.header("🔮 Predict CLV")

    recency = st.number_input("Recency", 0, 365, 30)
    frequency = st.number_input("Frequency", 1, 100, 5)
    monetary = st.number_input("Monetary", 0.0, 10000.0, 500.0)

    if st.button("Predict"):

        df = pd.DataFrame({
            "Recency":[10,20,5,30],
            "Frequency":[5,3,10,2],
            "Monetary":[500,300,1000,200],
            "CLV":[1200,700,2500,400]
        })

        X = df[["Recency","Frequency","Monetary"]]
        y = df["CLV"]

        model = GradientBoostingRegressor().fit(X, y)

        user = pd.DataFrame({
            "Recency":[recency],
            "Frequency":[frequency],
            "Monetary":[monetary]
        })

        pred = model.predict(user)

        file = "user_inputs.xlsx"

        try:
            old = pd.read_excel(file)
            pd.concat([old, user], ignore_index=True).to_excel(file, index=False)
        except:
            user.to_excel(file, index=False)

        st.success(f"💰 Predicted CLV: {pred[0]:.2f}")


# ---------------- SEGMENTATION ----------------
elif menu == "📊 Segmentation":

    st.header("📊 Customer Segmentation")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        r = st.selectbox("Recency column", df.columns)
        f = st.selectbox("Frequency column", df.columns)
        m = st.selectbox("Monetary column", df.columns)

        if st.button("Run Clustering"):

            X = df[[r,f,m]]
            X_scaled = StandardScaler().fit_transform(X)

            df["Cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

            st.dataframe(df)

            fig, ax = plt.subplots()
            ax.scatter(df[r], df[m], c=df["Cluster"])
            ax.set_xlabel("Recency")
            ax.set_ylabel("Monetary")

            st.pyplot(fig)


# ---------------- ADMIN PANEL ----------------
elif menu == "🔐 Admin Panel":

    st.header("🔐 Admin Panel")

    # -------- SESSION --------
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0

    if "blocked" not in st.session_state:
        st.session_state.blocked = False

    if "otp" not in st.session_state:
        st.session_state.otp = None

    if "otp_sent" not in st.session_state:
        st.session_state.otp_sent = False

    if "reset_mode" not in st.session_state:
        st.session_state.reset_mode = False

    saved_password = load_password()

    # -------- LOGIN --------
    if not st.session_state.blocked:

        password = st.text_input("Enter Password", type="password")

        if st.button("Login"):

            if password == saved_password:
                st.success("✅ Access Granted")
                st.session_state.attempts = 0

                st.subheader("📂 Server Files")
                st.write(os.listdir())

                try:
                    df = pd.read_excel("user_inputs.xlsx")
                    st.subheader("📄 Stored Data")
                    st.dataframe(df)
                except:
                    st.warning("No data found")

            else:
                st.session_state.attempts += 1
                left = 3 - st.session_state.attempts

                if left > 0:
                    st.error(f"❌ Wrong password! Attempts left: {left}")
                else:
                    st.session_state.blocked = True
                    st.session_state.reset_mode = True
                    st.error("🚫 Too many attempts. Reset required.")
                    st.rerun()

    # -------- RESET PASSWORD FLOW --------
    if st.session_state.reset_mode:

        st.warning("🔑 Password Reset")

        phone = st.text_input("Enter Mobile Number")

        if st.button("Send OTP"):

            otp = generate_otp()
            st.session_state.otp = otp
            st.session_state.otp_sent = True

            # Demo OTP display
            st.success(f"📲 OTP Sent: {otp}")

        if st.session_state.otp_sent:

            user_otp = st.text_input("Enter OTP")

            if st.button("Verify OTP"):

                if str(user_otp) == str(st.session_state.otp):

                    st.success("✅ OTP Verified")

                    new_pass = st.text_input("Create New Password", type="password")

                    if st.button("Save Password"):

                        if new_pass.strip() != "":
                            save_password(new_pass)

                            st.session_state.blocked = False
                            st.session_state.reset_mode = False
                            st.session_state.otp_sent = False
                            st.session_state.attempts = 0

                            st.success("🎉 Password Reset Successful! Login again.")
                            st.rerun()

                        else:
                            st.warning("Enter valid password")

                else:
                    st.error("❌ Invalid OTP")
