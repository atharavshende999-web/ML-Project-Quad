import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CLV Dashboard",
    layout="wide"
)

st.title("📊 Customer Lifetime Value (CLV) App")


# ---------------- SIDEBAR ----------------

menu = st.sidebar.radio(
    "Select Mode",
    [
        "🏠 Home",
        "🔮 CLV Predictor",
        "📊 Segmentation Dashboard"
    ]
)


# ---------------- HOME ----------------

if menu == "🏠 Home":

    st.header("Project Overview")

    st.write("""
    This application performs Customer Lifetime Value Prediction 
    and Customer Segmentation.

    ✔ Predict CLV using Machine Learning  
    ✔ Store user prediction data (hidden from users)  
    ✔ Segment customers using KMeans  
    ✔ Visualize clusters  
    """)


# ---------------- CLV PREDICTOR ----------------

elif menu == "🔮 CLV Predictor":

    st.header("🔮 Predict Customer Lifetime Value")

    st.subheader("Enter Customer RFM Values")

    recency = st.number_input("Recency (Days)", 0, 365, 30)
    frequency = st.number_input("Frequency (Purchases)", 1, 100, 5)
    monetary = st.number_input("Monetary Value ($)", 0.0, 10000.0, 500.0)

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

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        # User input
        user_data = pd.DataFrame({
            "Recency":[recency],
            "Frequency":[frequency],
            "Monetary":[monetary]
        })

        prediction = model.predict(user_data)

        # Add prediction
        user_data["Predicted_CLV"] = prediction[0]

        # -------- SAVE DATA IN EXCEL (HIDDEN FROM USER) --------

        file_name = "user_inputs.xlsx"

        try:
            old_data = pd.read_excel(file_name)

            final_data = pd.concat(
                [old_data, user_data],
                ignore_index=True
            )

            final_data.to_excel(
                file_name,
                index=False
            )

        except FileNotFoundError:
            user_data.to_excel(
                file_name,
                index=False
            )

        # -------- SHOW ONLY RESULT --------

        st.success(f"💰 Predicted CLV: ${prediction[0]:.2f}")
        st.success("✅ Prediction completed successfully")


# ---------------- SEGMENTATION ----------------

elif menu == "📊 Segmentation Dashboard":

    st.header("📊 Customer Segmentation Dashboard")

    file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        r_col = st.selectbox("Select Recency Column", df.columns)
        f_col = st.selectbox("Select Frequency Column", df.columns)
        m_col = st.selectbox("Select Monetary Column", df.columns)

        if st.button("Run Segmentation"):

            X = df[[r_col, f_col, m_col]]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=3, random_state=42)
            df["Cluster"] = kmeans.fit_predict(X_scaled)

            st.subheader("Clustered Customers")
            st.dataframe(df.head())

            fig, ax = plt.subplots()

            ax.scatter(
                df[r_col],
                df[m_col],
                c=df["Cluster"]
            )

            ax.set_xlabel("Recency")
            ax.set_ylabel("Monetary")
            ax.set_title("Customer Segmentation")

            st.pyplot(fig)

            st.success("✅ Segmentation Completed")
