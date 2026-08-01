import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

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
        users = {
            "host": {"password": "admin123"},
            "guests": {
                "Atharv": {"password": "Pass@123", "blocked": False},
                "Suraj": {"password": "Pass@123", "blocked": False},
                "Aryan": {"password": "Pass@123", "blocked": False},
                "Swaraj": {"password": "Pass@123", "blocked": False}
            }
        }
        with open(USER_FILE, "w") as f:
            json.dump(users, f)

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


# ---------------- SESSION ----------------
if "attempts" not in st.session_state:
    st.session_state.attempts = {}

if "host_logged_in" not in st.session_state:
    st.session_state.host_logged_in = False


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
    ✔ Multi-user login system  
    ✔ Admin approval system  
    ✔ Secure blocking mechanism  
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

    username = st.selectbox("Select Username", ["Atharv", "Suraj", "Aryan", "Swaraj"])
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):

        users = load_users()

        guest = users["guests"][username]

        # Track attempts per user
        if username not in st.session_state.attempts:
            st.session_state.attempts[username] = 0

        if guest["blocked"]:
            st.error("🚫 You are blocked")

            reqs = load_requests()
            if not any(r["user"] == username and r["status"] == "pending" for r in reqs):
                reqs.append({"user": username, "status": "pending"})
                save_requests(reqs)
                st.warning("Request sent to Admin")

        else:
            if pwd == guest["password"]:
                st.success(f"✅ Welcome {username}")
                st.session_state.attempts[username] = 0

            else:
                st.session_state.attempts[username] += 1

                if st.session_state.attempts[username] < 3:
                    st.error("❌ Password is wrong")
                else:
                    guest["blocked"] = True
                    save_users(users)

                    reqs = load_requests()
                    reqs.append({"user": username, "status": "pending"})
                    save_requests(reqs)

                    st.error("🚫 Blocked after 3 attempts")


# ==================================================
# 🧑‍💼 ADMIN PANEL
# ==================================================
elif menu == "🧑‍💼 Admin Panel":

    st.header("🧑‍💼 Admin Login")

    pwd = st.text_input("Enter Admin Password", type="password")

    if st.button("Login"):

        users = load_users()

        if pwd == users["host"]["password"]:
            st.session_state.host_logged_in = True
        else:
            st.error("❌ Password is wrong")

    if st.session_state.host_logged_in:

        st.success("✅ Admin Logged In")

        reqs = load_requests()
        users = load_users()

        if len(reqs) == 0:
            st.info("No requests")

        else:
            for i, r in enumerate(reqs):

                if r["status"] == "pending":

                    username = r["user"]

                    st.write(f"🔴 User: {username}")

                    new_pass = st.text_input(
                        f"New password for {username}",
                        key=f"pass_{i}",
                        type="password"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"Approve {i}"):

                            if new_pass.strip() == "":
                                st.error("Enter password")
                            else:
                                users["guests"][username]["password"] = new_pass
                                users["guests"][username]["blocked"] = False

                                save_users(users)

                                reqs[i]["status"] = "approved"
                                save_requests(reqs)

                                st.success(f"{username} unblocked")
                                st.rerun()

                    with col2:
                        if st.button(f"Reject {i}"):

                            reqs[i]["status"] = "rejected"
                            save_requests(reqs)

                            st.warning("Rejected")
                            st.rerun()
