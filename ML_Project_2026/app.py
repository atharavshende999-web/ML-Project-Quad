import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="CLV Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)


# ================= CUSTOM CSS =================

st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}


[data-testid="stSidebar"]{
    background-color:#111827;
}


[data-testid="stSidebar"] *{
    color:white;
}


h1{
    color:#1f2937;
}


.card{

background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 5px 20px #ddd;

}


.metric-card{

background:#2563eb;
color:white;
padding:20px;
border-radius:15px;
text-align:center;

}


button{

border-radius:10px !important;

}


</style>
""",unsafe_allow_html=True)



# ================= TITLE =================


st.markdown(
"""
<h1 style='text-align:center'>
📊 Customer Lifetime Value Analytics
</h1>

<p style='text-align:center'>
AI Powered Customer Prediction & Segmentation System
</p>
""",
unsafe_allow_html=True
)



# ================= SIDEBAR =================


st.sidebar.image(
"https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
width=100
)


st.sidebar.title(
"Navigation"
)


menu = st.sidebar.radio(
"",
[
"🏠 Dashboard",
"🔮 CLV Prediction",
"📊 Customer Segmentation",
"🔐 Admin Panel"
]
)



# ================= HOME =================


if menu=="🏠 Dashboard":


    st.markdown(
    """
    <div class="card">

    ## 🚀 Project Overview

    This application uses Machine Learning to:

    ✔ Predict Customer Lifetime Value  
    ✔ Analyze customer behaviour  
    ✔ Segment customers using KMeans  
    ✔ Store prediction history securely  

    </div>

    """,
    unsafe_allow_html=True
    )


    col1,col2,col3 = st.columns(3)


    with col1:
        st.markdown(
        """
        <div class="metric-card">

        🤖
        
        <h3>ML Model</h3>

        Gradient Boosting

        </div>
        """,
        unsafe_allow_html=True
        )


    with col2:

        st.markdown(
        """
        <div class="metric-card">

        📈

        <h3>Prediction</h3>

        CLV Forecast

        </div>
        """,
        unsafe_allow_html=True
        )


    with col3:

        st.markdown(
        """
        <div class="metric-card">

        👥

        <h3>Segmentation</h3>

        KMeans

        </div>
        """,
        unsafe_allow_html=True
        )



# ================= CLV PREDICTION =================


elif menu=="🔮 CLV Prediction":


    st.header("🔮 Customer Lifetime Value Prediction")


    col1,col2,col3=st.columns(3)


    with col1:
        recency=st.number_input(
        "📅 Recency",
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
        "💰 Monetary",
        0.0,
        10000.0,
        500.0
        )



    if st.button(
        "🚀 Predict CLV",
        use_container_width=True
    ):


        data=pd.DataFrame({

        "Recency":[10,20,5,30],
        "Frequency":[5,3,10,2],
        "Monetary":[500,300,1000,200],
        "CLV":[1200,700,2500,400]

        })


        model=GradientBoostingRegressor()

        model.fit(
        data[
        [
        "Recency",
        "Frequency",
        "Monetary"
        ]
        ],
        data["CLV"]
        )


        user=pd.DataFrame({

        "Recency":[recency],
        "Frequency":[frequency],
        "Monetary":[monetary]

        })


        result=model.predict(user)


        st.markdown(
        f"""
        <div class="metric-card">

        <h2>
        💰 Predicted CLV
        </h2>

        <h1>
        ${result[0]:.2f}
        </h1>

        </div>
        """,
        unsafe_allow_html=True
        )



        user["Predicted_CLV"]=result[0]


        file="user_inputs.xlsx"


        try:

            old=pd.read_excel(file)

            new=pd.concat(
            [old,user],
            ignore_index=True
            )

            new.to_excel(
            file,
            index=False
            )

        except:

            user.to_excel(
            file,
            index=False
            )



        st.success(
        "Customer data saved securely"
        )



# ================= SEGMENTATION =================


elif menu=="📊 Customer Segmentation":


    st.header(
    "📊 Customer Segmentation"
    )


    file=st.file_uploader(
    "Upload Customer Dataset",
    type="csv"
    )


    if file:


        df=pd.read_csv(file)


        st.dataframe(
        df.head(),
        use_container_width=True
        )


        r=st.selectbox(
        "Recency Column",
        df.columns
        )


        f=st.selectbox(
        "Frequency Column",
        df.columns
        )


        m=st.selectbox(
        "Monetary Column",
        df.columns
        )



        if st.button(
        "Create Segments"
        ):


            X=df[
            [
            r,f,m
            ]
            ]


            scaler=StandardScaler()

            X_scaled=scaler.fit_transform(X)


            model=KMeans(
            n_clusters=3,
            random_state=42
            )


            df["Cluster"]=model.fit_predict(
            X_scaled
            )


            st.dataframe(
            df,
            use_container_width=True
            )


            fig,ax=plt.subplots()

            ax.scatter(
            df[r],
            df[m],
            c=df["Cluster"]
            )


            ax.set_xlabel(
            "Recency"
            )

            ax.set_ylabel(
            "Monetary"
            )


            st.pyplot(fig)



# ================= ADMIN =================


elif menu=="🔐 Admin Panel":


    st.header(
    "🔐 Secure Admin Login"
    )


    username=st.text_input(
    "Username"
    )


    password=st.text_input(
    "Password",
    type="password"
    )


    users={
    "Atharv":"Pass@123",
    "Suraj":"Pass@123",
    "Aryan":"Pass@123",
    "Swaraj":"Pass@123"
    }



    if st.button(
    "Login",
    use_container_width=True
    ):


        if username in users and users[username]==password:


            st.success(
            "Welcome Admin"
            )


            try:

                df=pd.read_excel(
                "user_inputs.xlsx"
                )


                st.dataframe(
                df,
                use_container_width=True
                )


            except:

                st.warning(
                "No user data found"
                )


        else:

            st.error(
            "❌ Wrong Username or Password"
            )
