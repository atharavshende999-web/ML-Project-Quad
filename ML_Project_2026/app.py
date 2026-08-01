import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error



st.set_page_config(
    page_title="CLV Dashboard",
    layout="wide"
)

st.title("📊 Customer Lifetime Value (CLV) App")


menu = st.sidebar.radio(
    "Select Mode",
    ["🏠 Home", "🔮 CLV Predictor", "📊 Segmentation Dashboard"]
)



if menu == "🏠 Home":

    st.header("Project Overview")

    st.write("""
    This app is based on **Customer Lifetime Value Prediction and Segmentation**.

    ✔ Predict future customer revenue using Gradient Boosting  
    ✔ Segment customers using RFM + KMeans  
    ✔ Visualize customer groups
    """)


elif menu == "🔮 CLV Predictor":

    st.header("🔮 Predict Customer Lifetime Value")

    st.subheader("Enter Customer RFM Values")


    recency = st.number_input(
        "Recency (days)",
        0,
        365,
        30
    )


    frequency = st.number_input(
        "Frequency (purchases)",
        1,
        100,
        5
    )


    monetary = st.number_input(
        "Monetary Value ($)",
        0.0,
        10000.0,
        500.0
    )


    if st.button("Predict CLV"):

        data = {

            "Recency":[
                10,20,5,30,15,
                40,25,8,60,12,
                35,18
         ],

            "Frequency":[
                5,3,10,2,7,
                1,4,12,2,8,
                3,6
            ],

            "Monetary":[
                500,300,1000,200,700,
                100,400,1500,250,900,
                350,650
            ],

            "CLV":[
                1200,700,2500,400,1600,
                200,900,3000,500,2000,
                800,1400
            ]

        }


        df = pd.DataFrame(data) 


        X = df[
            [
            "Recency",
            "Frequency",
            "Monetary"
            ]
        ]


        y = df["CLV"]




        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )


        

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )


        model.fit(
            X_train,
            y_train
        )


        # Prediction

        input_data = pd.DataFrame(
            [
            [
            recency,
            frequency,
            monetary
            
            ]
            ],
            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]
        )


        prediction = model.predict(input_data)


       

        st.success(
            f"💰 Predicted CLV: ${prediction[0]:.2f}"
        )


    


elif menu == "📊 Segmentation Dashboard":


    st.header("📊 Customer Segmentation Dashboard")


    file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )


    if file:


        df = pd.read_csv(file)


        st.subheader("📌 Data Preview")

        st.dataframe(df.head())


        r_col = st.selectbox(
            "Select Recency Column",
            df.columns
        )


        f_col = st.selectbox(
            "Select Frequency Column",
            df.columns
        )


        m_col = st.selectbox(
            "Select Monetary Column",
            df.columns
        )



        if st.button("Run Segmentation"):


            X = df[
                [
                r_col,
                f_col,
                m_col
                ]
            ]


            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(X)



            kmeans = KMeans(
                n_clusters=3,
                random_state=42
            )


            df["Cluster"] = kmeans.fit_predict(
                X_scaled
            )


            st.subheader(
                "📊 Clustered Customers"
            )


            st.dataframe(
                df.head()
            )


            fig, ax = plt.subplots()


            ax.scatter(
                df[r_col],
                df[m_col],
                c=df["Cluster"]
            )


            ax.set_xlabel(
                "Recency"
            )


            ax.set_ylabel(
                "Monetary"
            )


            st.pyplot(fig)


            st.success(
                "✅ Segmentation Completed"
            )