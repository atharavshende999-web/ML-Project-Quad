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
    page_title="CLV Dashboard",
    page_icon="📊",
    layout="wide"
)



# ================= CUSTOM CSS =================

st.markdown(
"""
<style>

body{
background:#f8fafc;
}


[data-testid="stSidebar"]{

background-color:#0b1f4d;

}


[data-testid="stSidebar"] *{

color:white;

}


.card{

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 5px 15px #dddddd;

}


</style>

""",
unsafe_allow_html=True
)



# ================= TITLE =================

st.markdown(
"""
<h1 style="text-align:center;">
📊 Customer Lifetime Value Dashboard
</h1>

<p style="text-align:center;">
AI Powered Customer Prediction and Segmentation System
</p>

""",
unsafe_allow_html=True
)




# ================= SIDEBAR =================


with st.sidebar:


    if os.path.exists("robot.png"):

        st.image(
            "robot.png",
            width=120
        )

    else:

        st.write("🤖")


    st.title(
        "🤖 CLV Assistant"
    )


    menu = st.radio(
        "Select Mode",
        [
            "🏠 Home",
            "🔮 CLV Predictor",
            "📊 Segmentation Dashboard",
            "🔐 Admin Panel"
        ]
    )





# ================= HOME =================


if menu=="🏠 Home":


    st.markdown(
    """
    <div class="card">


    ## 🚀 Project Overview


    ✔ Predict Customer Lifetime Value using ML  


    ✔ Customer segmentation using KMeans  


    ✔ Store prediction history securely  


    ✔ Admin controlled data access  


    </div>

    """,
    unsafe_allow_html=True
    )





# ================= CLV PREDICTOR =================


elif menu=="🔮 CLV Predictor":


    st.header(
        "🔮 Predict Customer Lifetime Value"
    )


    col1,col2,col3 = st.columns(3)


    with col1:

        recency = st.number_input(
            "📅 Recency Days",
            0,
            365,
            30
        )


    with col2:

        frequency = st.number_input(
            "🛒 Frequency",
            1,
            100,
            5
        )


    with col3:

        monetary = st.number_input(
            "💰 Monetary Value",
            0.0,
            10000.0,
            500.0
        )




    if st.button(
        "🚀 Predict CLV",
        use_container_width=True
    ):



        data = {

            "Recency":[10,20,5,30,15,40,25,8],

            "Frequency":[5,3,10,2,7,1,4,12],

            "Monetary":[500,300,1000,200,700,100,400,1500],

            "CLV":[1200,700,2500,400,1600,200,900,3000]

        }



        df=pd.DataFrame(data)



        X=df[
            [
            "Recency",
            "Frequency",
            "Monetary"
            ]
        ]


        y=df["CLV"]



        X_train,X_test,y_train,y_test=train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )



        model=GradientBoostingRegressor(
            random_state=42
        )



        model.fit(
            X_train,
            y_train
        )



        user_data=pd.DataFrame({

            "Recency":[recency],

            "Frequency":[frequency],

            "Monetary":[monetary]

        })



        prediction=model.predict(
            user_data
        )



        user_data["Predicted_CLV"]=prediction[0]



        st.success(
            f"💰 Predicted CLV : ${prediction[0]:.2f}"
        )



        # SAVE EXCEL


        file="user_inputs.xlsx"



        if os.path.exists(file):


            old=pd.read_excel(file)


            final=pd.concat(
                [
                old,
                user_data
                ],
                ignore_index=True
            )


            final.to_excel(
                file,
                index=False
            )


        else:


            user_data.to_excel(
                file,
                index=False
            )



        st.success(
            "✅ Customer data stored"
        )





# ================= SEGMENTATION =================


elif menu=="📊 Segmentation Dashboard":


    st.header(
        "📊 Customer Segmentation"
    )


    file=st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )



    if file:


        df=pd.read_csv(file)


        st.dataframe(
            df.head()
        )



        r=st.selectbox(
            "Select Recency Column",
            df.columns
        )


        f=st.selectbox(
            "Select Frequency Column",
            df.columns
        )


        m=st.selectbox(
            "Select Monetary Column",
            df.columns
        )



        if st.button(
            "Create Segments"
        ):



            X=df[
                [
                r,
                f,
                m
                ]
            ]



            scaler=StandardScaler()


            X_scaled=scaler.fit_transform(
                X
            )



            kmeans=KMeans(
                n_clusters=3,
                random_state=42
            )



            df["Cluster"]=kmeans.fit_predict(
                X_scaled
            )



            st.dataframe(
                df
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


            ax.set_title(
                "Customer Segmentation"
            )


            st.pyplot(fig)





# ================= ADMIN PANEL =================


elif menu=="🔐 Admin Panel":


    st.header(
        "🔐 Admin Login"
    )



    users={

        "atharv":"Pass@123",

        "suraj":"Pass@123",

        "aryan":"Pass@123",

        "swaraj":"Pass@123"

    }



    username=st.text_input(
        "Enter Username"
    )


    password=st.text_input(
        "Enter Password",
        type="password"
    )



    if st.button(
        "Login"
    ):


        if username.lower() in users:


            if users[username.lower()] == password:


                st.success(
                    "✅ Login Successful"
                )



                try:


                    df=pd.read_excel(
                        "user_inputs.xlsx"
                    )


                    st.subheader(
                        "📄 Stored Customer Data"
                    )


                    st.dataframe(
                        df
                    )



                    with open(
                        "user_inputs.xlsx",
                        "rb"
                    ) as f:


                        st.download_button(

                            "📥 Download Excel",

                            f,

                            file_name="user_inputs.xlsx"

                        )


                except:


                    st.warning(
                        "No data available"
                    )



            else:

                st.error(
                    "❌ Password is wrong"
                )



        else:


            st.error(
                "❌ Username not found"
            )
