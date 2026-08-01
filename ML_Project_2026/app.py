import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
import smtplib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


# ---------------- CONFIG ----------------
st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value (CLV) App")

USER_FILE = "users.json"
REQUEST_FILE = "requests.json"


# ---------------- INIT FILES ----------------
def init_files():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({
                "host": {"password": "admin123"},
                "guest": {
                    "username": "guest",
                    "password": "guest123",
                    "blocked": False
                }
            }, f)

    if not os.path.exists(REQUEST_FILE):
        with open(REQUEST_FILE, "w") as f:
            json.dump([], f)

init_files()


# ---------------- LOAD/SAVE ----------------
def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USER_FILE, "w") as f:
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

    sender_email = st.secrets["EMAIL"]
    app_password = st.secrets["APP_PASSWORD"]

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

if "otp" not in st.session_state:
    st.session_state.otp = None

if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False

if "reset_mode" not in st.session_state:
    st.session_state.reset_mode = False

if "host_logged_in" not in st.session_state:
    st.session_state.host_logged_in = False


users = load_users()


# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Select Mode",
    ["🏠 Home", "🔮 CLV Predictor", "📊 Segmentation", "👤 Guest Login", "🧑‍💼 Admin Panel"]
)


# ==================================================
# 🏠 HOME
# ==================================================
if menu == "🏠 Home":
    st.header("Project Overview")
    st.write("""
    ✔ CLV Prediction using ML  
    ✔ Customer segmentation  
    ✔ Secure multi-role login  
    ✔ Host approval system  
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
            st.pyplot(fig)


# ==================================================
# 👤 GUEST LOGIN
# ==================================================
elif menu == "👤 Guest Login":

    st.header("👤 Guest Login")

    username = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):

        guest = users["guest"]

        if guest["blocked"]:
            st.error("🚫 You are blocked")

            reqs = load_requests()
            if not any(r["user"] == username and r["status"] == "pending" for r in reqs):
                reqs.append({"user": username, "status": "pending"})
                save_requests(reqs)
                st.warning("Request sent to Admin")

        else:
            if username == guest["username"] and pwd == guest["password"]:
                st.success("✅ Login Successful")
                st.session_state.attempts = 0

            else:
                st.session_state.attempts += 1
                left = 3 - st.session_state.attempts

                if left > 0:
                    st.error(f"❌ Wrong password. Attempts left: {left}")
                else:
                    guest["blocked"] = True
                    save_users(users)

                    reqs = load_requests()
                    reqs.append({"user": username, "status": "pending"})
                    save_requests(reqs)

                    st.error("🚫 Blocked! Request sent to Admin")


# ==================================================
# 🧑‍💼 ADMIN PANEL
# ==================================================
elif menu == "🧑‍💼 Admin Panel":

    st.header("🧑‍💼 Admin Login")

    pwd = st.text_input("Enter Admin Password", type="password")

    if st.button("Login"):

        if pwd == users["host"]["password"]:
            st.session_state.host_logged_in = True
        else:
            st.error("Wrong password")

    if st.session_state.host_logged_in:

        st.success("✅ Admin Logged In")

        reqs = load_requests()

        if len(reqs) == 0:
            st.info("No requests")

        else:
            for i, r in enumerate(reqs):

                if r["status"] == "pending":

                    st.write(f"🔴 User: {r['user']}")

                    new_pass = st.text_input(
                        f"New password for {r['user']}",
                        key=f"pass_{i}",
                        type="password"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"Approve {i}"):

                            if new_pass.strip() == "":
                                st.error("Enter password")
                            else:
                                users["guest"]["password"] = new_pass
                                users["guest"]["blocked"] = False
                                save_users(users)

                                reqs[i]["status"] = "approved"
                                save_requests(reqs)

                                st.success("User unblocked")
                                st.rerun()

                    with col2:
                        if st.button(f"Reject {i}"):

                            reqs[i]["status"] = "rejected"
                            save_requests(reqs)

                            st.warning("Rejected")
                            st.rerun()
