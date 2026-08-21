st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #F5F7FA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A8A, #312E81);
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    border-left: 5px solid #2563EB;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
    font-weight: bold;
    border: none;
}

/* Inputs */
.stNumberInput, .stTextInput {
    background: white;
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Headers */
h1, h2, h3 {
    color: #111827;
}

/* Success Box */
.stSuccess {
    border-radius: 10px;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)
