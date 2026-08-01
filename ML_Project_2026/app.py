import streamlit as st
import json
import os
import random
import smtplib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Secure Role System", layout="wide")
st.title("🔐 Role-Based Login System")

CRED_FILE = "credentials.json"
REQ_FILE = "requests.json"


# ---------------- INIT FILES ----------------
def init_files():
    if not os.path.exists(CRED_FILE):
        with open(CRED_FILE, "w") as f:
            json.dump({
                "host_password": "admin123",
                "guest_password": "guest123"
            }, f)

    if not os.path.exists(REQ_FILE):
        with open(REQ_FILE, "w") as f:
            json.dump([], f)

init_files()


# ---------------- LOAD/SAVE ----------------
def load_creds():
    with open(CRED_FILE) as f:
        return json.load(f)

def load_requests():
    with open(REQ_FILE) as f:
        return json.load(f)

def save_requests(data):
    with open(REQ_FILE, "w") as f:
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


# ---------------- ROLE ----------------
role = st.sidebar.selectbox("Login As", ["Guest", "Host"])
creds = load_creds()


# ==================================================
# 👤 GUEST LOGIN
# ==================================================
if role == "Guest":

    st.header("👤 Guest Login")

    if not st.session_state.blocked:

        pwd = st.text_input("Enter Guest Password", type="password")

        if st.button("Login"):

            if pwd == creds["guest_password"]:
                st.success("✅ Guest Login Successful")
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
        st.error("🚫 You are blocked.")

        if st.button("🔑 Reset via Email OTP"):
            st.session_state.reset_mode = True


# ==================================================
# 🔐 EMAIL RESET
# ==================================================
if st.session_state.reset_mode:

    st.warning("🔑 Reset Password via Email OTP")

    email = st.text_input("Enter your Email")

    if st.button("Send OTP"):
        try:
            otp = send_email_otp(email)
            st.session_state.otp = otp
            st.session_state.otp_sent = True
            st.success("📧 OTP sent")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.otp_sent:

        user_otp = st.text_input("Enter OTP")

        if st.button("Verify OTP"):
            if str(user_otp) == str(st.session_state.otp):

                st.success("✅ OTP Verified")

                new_pass = st.text_input("New Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")

                if st.button("Save Password"):
                    if new_pass == confirm and new_pass.strip() != "":
                        creds["guest_password"] = new_pass

                        with open(CRED_FILE, "w") as f:
                            json.dump(creds, f)

                        st.session_state.blocked = False
                        st.session_state.attempts = 0
                        st.session_state.reset_mode = False
                        st.session_state.otp_sent = False

                        st.success("🎉 Password Reset Successful")
                        st.rerun()
                    else:
                        st.error("Passwords do not match")
            else:
                st.error("Invalid OTP")


# ==================================================
# 🧑‍💼 HOST LOGIN
# ==================================================
if role == "Host":

    st.header("🧑‍💼 Host Login")

    pwd = st.text_input("Enter Host Password", type="password")

    if st.button("Login as Host"):

        if pwd == creds["host_password"]:

            st.success("✅ Host Logged In")

            st.subheader("📨 Guest Block Requests")

            requests = load_requests()

            if len(requests) == 0:
                st.info("No pending requests")

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

                                st.success("✅ Guest Unblocked")
                                st.rerun()

                        with col2:
                            if st.button(f"Reject {i}"):
                                requests[i]["status"] = "rejected"
                                save_requests(requests)

                                st.warning("❌ Request Rejected")
                                st.rerun()

        else:
            st.error("❌ Wrong Host Password")
