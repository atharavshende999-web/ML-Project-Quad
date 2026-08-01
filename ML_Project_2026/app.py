import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
import smtplib
import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


# ---------------- CONFIG ----------------
st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value (CLV) App")

PASSWORD_FILE = "password.json"
REQUEST_FILE = "requests.json"


# ---------------- INIT FILES ----------------
def init_files():
    if not os.path.exists(PASSWORD_FILE):
        with open(PASSWORD_FILE, "w") as f:
            json.dump({
                "host_password": "admin123",
                "guest_password": "guest123"
            }, f)

    if not os.path.exists(REQUEST_FILE):
        with open(REQUEST_FILE, "w") as f:
            json.dump([], f)

init_files()


# ---------------- LOAD/SAVE ----------------
def load_passwords():
    with open(PASSWORD_FILE, "r") as f:
        return json.load(f)

def save_passwords(data):
    with open(PASSWORD_FILE, "w") as f:
        json.dump(data, f)

def load_requests():
    with open(REQUEST_FILE, "r") as f:
        return json.load(f)

def save_requests(data):
    with open(REQUEST_FILE, "w") as f:
        json.dump(data, f)


# ---------------- EMAIL OTP ----------------
def send_email_otp(receiver_email):
    otp = random.randint(100000, 999999)

    sender_email = "your_email@gmail.com"
    app_password = "your_app_password"

    message = f"Subject: OTP\n\nYour OTP is {otp}"

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, message)
    server.quit()

    return otp


# ---------------- SESSION ----------------
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


passwords = load_passwords()


# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Select Mode",
    ["🏠 Home", "🔮 CLV Predictor", "📊 Segmentation", "👤 Guest Login", "🧑‍💼 Host Panel"]
)


# ==================================================
# 🏠 HOME
# ==================================================
if menu == "🏠 Home":
    st.header("Project Overview")
    st.write("""
    ✔ CLV Prediction using ML  
    ✔ Customer segmentation  
    ✔ Secure login system  
    ✔ Host approval for blocked users  
    ✔ Email OTP reset  
    """)


# ==================================================
# 🔮 CLV PREDICTOR
# ==================================================
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


# ==================================================
# 📊 SEGMENTATION
# ==================================================
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


# ==================================================
# 👤 GUEST LOGIN
# ==================================================
elif menu == "👤 Guest Login":

    st.header("👤 Guest Login")

    if not st.session_state.blocked:

        pwd = st.text_input("Enter Guest Password", type="password")

        if st.button("Login"):

            if pwd == passwords["guest_password"]:
                st.success("✅ Login Success")
                st.session_state.attempts = 0

            else:
                st.session_state.attempts += 1
                left = 3 - st.session_state.attempts

                if left > 0:
                    st.error(f"❌ Wrong password. Attempts left: {left}")
                else:
                    st.session_state.blocked = True

                    reqs = load_requests()
                    reqs.append({"status": "pending"})
                    save_requests(reqs)

                    st.error("🚫 Blocked! Request sent to Host.")

    else:
        st.error("🚫 You are blocked")

        if st.button("🔑 Reset via Email OTP"):
            st.session_state.reset_mode = True


# ==================================================
# 🔑 OTP RESET
# ==================================================
if st.session_state.reset_mode:

    st.warning("🔑 Reset Password")

    email = st.text_input("Enter Email")

    if st.button("Send OTP"):
        try:
            otp = send_email_otp(email)
            st.session_state.otp = otp
            st.session_state.otp_sent = True
            st.success("OTP Sent")
        except Exception as e:
            st.error(e)

    if st.session_state.otp_sent:

        user_otp = st.text_input("Enter OTP")

        if st.button("Verify OTP"):

            if str(user_otp) == str(st.session_state.otp):

                new_pass = st.text_input("New Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")

                if st.button("Save"):

                    if new_pass == confirm:
                        passwords["guest_password"] = new_pass
                        save_passwords(passwords)

                        st.session_state.blocked = False
                        st.session_state.attempts = 0
                        st.session_state.reset_mode = False

                        st.success("Password Reset Done")
                        st.rerun()
                    else:
                        st.error("Passwords not match")

            else:
                st.error("Invalid OTP")


# ==================================================
# 🧑‍💼 HOST PANEL
# ==================================================
elif menu == "🧑‍💼 Host Panel":

    st.header("🧑‍💼 Host Login")

    pwd = st.text_input("Enter Host Password", type="password")

    if st.button("Login as Host"):

        if pwd == passwords["host_password"]:

            st.success("✅ Host Logged In")

            requests = load_requests()

            st.subheader("📨 Block Requests")

            if len(requests) == 0:
                st.info("No requests")

            else:
                for i, r in enumerate(requests):

                    if r["status"] == "pending":

                        st.write(f"Request #{i}")

                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button(f"Approve {i}"):
                                requests[i]["status"] = "approved"
                                save_requests(requests)

                                st.session_state.blocked = False
                                st.session_state.attempts = 0

                                st.success("User Unblocked")
                                st.rerun()

                        with col2:
                            if st.button(f"Reject {i}"):
                                requests[i]["status"] = "rejected"
                                save_requests(requests)

                                st.warning("Rejected")
                                st.rerun()

        else:
            st.error("Wrong Host Password")
