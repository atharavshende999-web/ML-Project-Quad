# app.py
# Complete Streamlit CLV Dashboard template

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CLV Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0b1f4d;}
[data-testid="stSidebar"] *{color:white;}
.card{background:white;padding:20px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);}
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=110)
menu = st.sidebar.radio("Navigation",
["🏠 Home","🔮 CLV Predictor","📊 Segmentation Dashboard","🔐 Admin Panel"])

st.title("📊 Customer Lifetime Value Dashboard")

if menu=="🏠 Home":
    st.markdown('<div class="card"><h2>Welcome</h2><p>Predict CLV, segment customers and manage data.</p></div>', unsafe_allow_html=True)

elif menu=="🔮 CLV Predictor":
    c1,c2,c3=st.columns(3)
    with c1:
        recency=st.number_input("Recency",0,365,30)
    with c2:
        frequency=st.number_input("Frequency",1,100,5)
    with c3:
        monetary=st.number_input("Monetary",0.0,10000.0,500.0)
    if st.button("Predict CLV"):
        df=pd.DataFrame({
            "Recency":[10,20,5,30,15,40,25,8],
            "Frequency":[5,3,10,2,7,1,4,12],
            "Monetary":[500,300,1000,200,700,100,400,1500],
            "CLV":[1200,700,2500,400,1600,200,900,3000]
        })
        X=df[["Recency","Frequency","Monetary"]]
        y=df["CLV"]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
        model=GradientBoostingRegressor(random_state=42)
        model.fit(Xtr,ytr)
        user=pd.DataFrame({"Recency":[recency],"Frequency":[frequency],"Monetary":[monetary]})
        pred=model.predict(user)[0]
        st.success(f"Predicted CLV: ${pred:.2f}")
        user["Predicted_CLV"]=pred
        f="user_inputs.xlsx"
        if os.path.exists(f):
            old=pd.read_excel(f)
            pd.concat([old,user],ignore_index=True).to_excel(f,index=False)
        else:
            user.to_excel(f,index=False)

elif menu=="📊 Segmentation Dashboard":
    up=st.file_uploader("Upload CSV",type=["csv"])
    if up:
        df=pd.read_csv(up)
        st.dataframe(df.head())
        r=st.selectbox("Recency",df.columns)
        f=st.selectbox("Frequency",df.columns)
        m=st.selectbox("Monetary",df.columns)
        if st.button("Run Segmentation"):
            X=StandardScaler().fit_transform(df[[r,f,m]])
            df["Cluster"]=KMeans(n_clusters=3,random_state=42).fit_predict(X)
            st.dataframe(df)
            fig,ax=plt.subplots()
            ax.scatter(df[r],df[m],c=df["Cluster"])
            st.pyplot(fig)

elif menu=="🔐 Admin Panel":
    users={"atharv":"Pass@123","suraj":"Pass@123","aryan":"Pass@123","swaraj":"Pass@123"}
    u=st.text_input("Username")
    p=st.text_input("Password",type="password")
    if st.button("Login"):
        if u.lower() in users and users[u.lower()]==p:
            st.success("Login successful")
            if os.path.exists("user_inputs.xlsx"):
                df=pd.read_excel("user_inputs.xlsx")
                st.dataframe(df)
                with open("user_inputs.xlsx","rb") as fh:
                    st.download_button("Download Excel",fh,file_name="user_inputs.xlsx")
            else:
                st.info("No stored data.")
        else:
            st.error("Wrong username or password")
