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

from twilio.rest import Client


# ---------------- CONFIG ----------------
st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value (CLV) App")

PASSWORD_FILE = "password.json"

TWILIO_SID = "YOUR_SID"
TWILIO_AUTH = "YOUR_AUTH"
TWILIO_PHONE = "YOUR_PHONE"


# ---------------- PASSWORD STORAGE ----------------
def load_password():
    if os.path.exists(PASSWORD_FILE):
        with open(PASSWORD_FILE, "r") as f:
            return json.load(f)["password"]
    return "admin123"

def save_password(new_pass):
    with open(PASSWORD_FILE, "w") as f:
        json.dump({"password": new_pass}, f)


# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Select Mode",
    ["🏠 Home", "🔮 CLV Predictor", "📊 Segmentation", "🔐 Admin Panel"]
)


# ---------------- HOME ----------------
if menu == "🏠 Home":
    st.header("Project Overview")
    st.write("""
    ✔ ML CLV Prediction  
    ✔ Data storage  
    ✔ Segmentation  
    ✔ OTP Login + Password Reset  
    """)


# ---------------- CLV ----------------
elif menu == "🔮 CLV Predictor":

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

        st.success(f"💰 CLV: {pred[0]:.2f}")


# ---------------- SEGMENTATION ----------------
elif menu == "📊 Segmentation":

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        r = st.selectbox("Recency", df.columns)
        f = st.selectbox("Frequency", df.columns)
        m = st.selectbox("Monetary", df.columns)

        if st.button("Cluster"):

            X = df[[r,f,m]]
            X_scaled = StandardScaler().fit_transform(X)

            df["Cluster"] = KMeans(n_clusters=3).fit_predict(X_scaled)

            st.dataframe(df)

            fig, ax = plt.subplots()
            ax.scatter(df[r], df[m], c=df["Cluster"])
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

                st.write("📂 Files:", os.listdir())

                try:
                    df = pd.read_excel("user_inputs.xlsx")
                    st.dataframe(df)
                except:
                    st.warning("No data")

            else:
                st.session_state.attempts += 1
                left = 3 - st.session_state.attempts

                if left > 0:
                    st.error(f"❌ Wrong password! Left: {left}")
                else:
                    st.session_state.blocked = True
                    st.session_state.reset_mode = True
                    st.error("🚫 Blocked. Reset required.")
                    st.rerun()

    # -------- RESET FLOW --------
    if st.session_state.reset_mode:

        st.warning("🔁 Reset Password")

        phone = st.text_input("Enter Phone (+countrycode)")

        if st.button("Send OTP"):

            otp = random.randint(100000,999999)
            st.session_state.otp = otp
            st.session_state.otp_sent = True

            client = Client(TWILIO_SID, TWILIO_AUTH)

            try:
                client.messages.create(
                    body=f"Your OTP is {otp}",
                    from_=TWILIO_PHONE,
                    to=phone
                )
                st.success("📲 OTP Sent")
            except:
                st.error("OTP failed")

        if st.session_state.otp_sent:

            user_otp = st.text_input("Enter OTP")

            if st.button("Verify OTP"):

                if str(user_otp) == str(st.session_state.otp):

                    st.success("✅ Verified")

                    new_pass = st.text_input("New Password", type="password")

                    if st.button("Save Password"):

                        save_password(new_pass)

                        st.session_state.blocked = False
                        st.session_state.reset_mode = False
                        st.session_state.otp_sent = False
                        st.session_state.attempts = 0

                        st.success("🎉 Password Reset Done")
                        st.rerun()

                else:
                    st.error("❌ Wrong OTP")
