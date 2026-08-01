import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="CLV Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)


# ================= CSS DESIGN =================

st.markdown("""
<style>

.main{
background-color:#f8fafc;
}


[data-testid="stSidebar"]{
background:#111827;
}


[data-testid="stSidebar"] *{
color:white;
}


.card{

background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 4px 15px #dddddd;

}


.title{

text-align:center;
font-size:40px;
font-weight:bold;

}


.subtitle{

text-align:center;
color:gray;
font-size:18px;

}


</style>

""",unsafe_allow_html=True)



# ================= TITLE =================


st.markdown(
"""
<div class="title">
📊 Customer Lifetime Value Analytics
</div>

<div class="subtitle">
AI Based Customer Prediction and Segmentation System
</div>

<br>
""",

unsafe_allow_html=True
)



# ================= SIDEBAR =================


st.sidebar.title("Navigation")


menu = st.sidebar.radio(
"",
[
"🏠 Home",
"🔮 CLV Prediction",
"📊 Customer Segmentation",
"🔐 Admin Panel"
]
)



# ================= HOME =================


if menu=="🏠 Home":


    st.markdown(
    """
    <div class="card">

    ## 🚀 Project Overview


    This application performs:


    ✔ Customer Lifetime Value Prediction  


    ✔ Machine Learning based Revenue Forecasting  


    ✔ Customer Segmentation using KMeans  


    ✔ Secure Admin Data Management  


    </div>

    """,
    unsafe_allow_html=True
    )


    st.write("")


    col1,col2,col3=st.columns(3)


    with col1:

        st.info(
        """
        🤖

        **Machine Learning**

        Gradient Boosting
        """
        )


    with col2:

        st.info(
        """
        📈

        **Prediction**

        Future CLV
        """
        )


    with col3:

        st.info(
        """
        👥

        **Segmentation**

        KMeans Clustering
        """
        )



# ================= CLV PREDICTION =================


elif menu=="🔮 CLV Prediction":


    st.header("🔮 Customer Lifetime Value Prediction")


    col1,col2,col3=st.columns(3)



    with col1:

        recency=st.number_input(
        "📅 Recency Days",
        0,
        365,
        30
        )


    with col2:

        frequency=st.number_input(
        "🛒 Frequency",
        1,
        100,
        5
        )


    with col3:

        monetary=st.number_input(
        "💰 Monetary Value",
        0.0,
        10000.0,
        500.0
        )



    if st.button(
    "🚀 Predict CLV",
    use_container_width=True
    ):



        training=pd.DataFrame({

        "Recency":[10,20,5,30,15,40],

        "Frequency":[5,3,10,2,7,1],

        "Monetary":[500,300,1000,200,700,100],

        "CLV":[1200,700,2500,400,1600,200]

        })



        model=GradientBoostingRegressor(
        random_state=42
        )


        model.fit(

        training[
        [
        "Recency",
        "Frequency",
        "Monetary"
        ]
        ],

        training["CLV"]

        )



        customer=pd.DataFrame({

        "Recency":[recency],

        "Frequency":[frequency],

        "Monetary":[monetary]

        })



        prediction=model.predict(customer)



        customer["Predicted_CLV"]=prediction[0]



        st.success(
        f"💰 Predicted CLV : ${prediction[0]:.2f}"
        )



        # SAVE EXCEL


        file="user_inputs.xlsx"



        if os.path.exists(file):

            old=pd.read_excel(file)

            new=pd.concat(
            [old,customer],
            ignore_index=True
            )

            new.to_excel(
            file,
            index=False
            )


        else:

            customer.to_excel(
            file,
            index=False
            )



        st.success(
        "✅ Customer prediction saved"
        )
