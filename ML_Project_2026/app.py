import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ============================================================
# CLV INTELLIGENCE — FINAL STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="CLV Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f6f8fc;
}


/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 55%,
        #172554 100%
    );

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.sidebar-brand {
    text-align: center;
    padding: 14px 8px 24px;
}

.sidebar-logo {
    width: 64px;
    height: 64px;
    margin: auto;

    border-radius: 18px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 32px;

    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6
    );

    box-shadow:
        0 12px 30px rgba(99,102,241,.35);
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    margin-top: 12px;
}

.sidebar-subtitle {
    color: #94a3b8 !important;
    font-size: 12px;
    margin-top: 3px;
}


/* ================= PAGE HEADINGS ================= */

.page-title {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;

    letter-spacing: -1px;

    margin-bottom: 2px;
}

.page-subtitle {
    color: #64748b;
    font-size: 15px;

    margin-bottom: 26px;
}

.section-title {
    font-size: 22px;
    font-weight: 800;

    color: #0f172a;

    margin: 22px 0 12px;
}


/* ================= CARDS ================= */

.card {

    background: #ffffff;

    border: 1px solid #e8edf5;

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 8px 30px rgba(15,23,42,.055);
}


/* ================= HERO ================= */

.hero {

    background:
        linear-gradient(
            135deg,
            #111827 0%,
            #312e81 52%,
            #4f46e5 100%
        );

    border-radius: 24px;

    padding: 34px;

    color: white;

    box-shadow:
        0 18px 45px rgba(49,46,129,.22);

    margin-bottom: 22px;
}

.hero h1 {

    color: white;

    font-size: 34px;

    margin:
        0
        0
        10px;
}

.hero p {

    color: #dbeafe;

    max-width: 760px;

    font-size: 15px;

    line-height: 1.7;
}

.hero-badge {

    display: inline-block;

    padding:
        7px
        12px;

    border-radius: 999px;

    background:
        rgba(255,255,255,.12);

    border:
        1px solid
        rgba(255,255,255,.16);

    font-size: 12px;

    margin-bottom: 12px;
}


/* ================= KPI CARDS ================= */

.kpi {

    background: white;

    border:
        1px solid
        #e8edf5;

    border-radius: 18px;

    padding: 20px;

    min-height: 130px;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,.05);
}

.kpi-icon {
    font-size: 25px;
}

.kpi-label {

    color: #64748b;

    font-size: 12px;

    font-weight: 700;

    margin-top: 9px;

    text-transform: uppercase;

    letter-spacing: .5px;
}

.kpi-value {

    color: #0f172a;

    font-size: 27px;

    font-weight: 800;

    margin-top: 3px;
}

.kpi-note {

    color: #94a3b8;

    font-size: 11px;

    margin-top: 3px;
}


/* ================= PREDICTION ================= */

.prediction {

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    border-radius: 22px;

    padding: 30px;

    color: white;

    text-align: center;

    box-shadow:
        0 18px 40px
        rgba(79,70,229,.24);

    margin: 20px 0;
}

.prediction-label {

    font-size: 13px;

    opacity: .85;

    font-weight: 600;
}

.prediction-value {

    font-size: 45px;

    font-weight: 800;

    margin:
        5px
        0;
}

.prediction-caption {

    font-size: 12px;

    opacity: .8;
}


/* ================= SEGMENT CARDS ================= */

.segment {

    border-radius: 16px;

    padding: 18px;

    background: white;

    border:
        1px solid
        #e8edf5;

    box-shadow:
        0 6px 20px
        rgba(15,23,42,.04);
}

.segment h3 {

    margin:
        0
        0
        5px;

    color: #0f172a;
}

.segment p {

    margin: 0;

    color: #64748b;

    font-size: 13px;
}


/* ================= BUTTONS ================= */

.stButton > button {

    border-radius: 11px;

    min-height: 44px;

    font-weight: 700;

    border:
        1px solid
        #e2e8f0;

    transition:
        all .2s ease;
}

.stButton > button:hover {

    transform:
        translateY(-1px);

    box-shadow:
        0 8px 20px
        rgba(15,23,42,.10);
}


/* ================= INPUTS ================= */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {

    border-radius: 10px !important;
}


/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {

    background: #fff;

    border:
        1px dashed
        #cbd5e1;

    border-radius: 15px;

    padding: 8px;
}


/* ================= STREAMLIT CLEANUP ================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ================= STATUS ================= */

.status-dot {

    display: inline-block;

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #22c55e;

    margin-right: 6px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def train_clv_model():

    data = {

        "Recency": [
            10, 20, 5, 30,
            15, 40, 25, 8,
            60, 12, 35, 18
        ],

        "Frequency": [
            5, 3, 10, 2,
            7, 1, 4, 12,
            2, 8, 3, 6
        ],

        "Monetary": [
            500, 300, 1000, 200,
            700, 100, 400, 1500,
            250, 900, 350, 650
        ],

        "CLV": [
            1200, 700, 2500, 400,
            1600, 200, 900, 3000,
            500, 2000, 800, 1400
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
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def save_prediction(user_data):

    file_name = "user_inputs.xlsx"

    try:

        old = pd.read_excel(
            file_name
        )

        new = pd.concat(
            [
                old,
                user_data
            ],

            ignore_index=True
        )

    except Exception:

        new = user_data

    new.to_excel(
        file_name,
        index=False
    )


# ============================================================
# SESSION STATE
# ============================================================

if "attempts" not in st.session_state:

    st.session_state.attempts = 0


if "blocked" not in st.session_state:

    st.session_state.blocked = False


if "admin_logged_in" not in st.session_state:

    st.session_state.admin_logged_in = False


if "last_prediction" not in st.session_state:

    st.session_state.last_prediction = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">

        <div class="sidebar-logo">
            📊
        </div>

        <div class="sidebar-title">
            CLV Intelligence
        </div>

        <div class="sidebar-subtitle">
            AI CUSTOMER ANALYTICS
        </div>

    </div>
    """, unsafe_allow_html=True)


    menu = st.radio(

        "NAVIGATION",

        [
            "🏠 Overview",
            "🔮 CLV Predictor",
            "👥 Customer Segmentation",
            "🔐 Admin Center"
        ],

        label_visibility="collapsed"
    )


    st.markdown("---")


    st.markdown("""
    <div style="
        padding:8px 4px;
        color:#cbd5e1 !important;
        font-size:12px;
    ">

        <span class="status-dot"></span>
        System Ready

        <br><br>

        <b>ML Engine</b>
        <br>
        Gradient Boosting

        <br><br>

        <b>Segmentation</b>
        <br>
        K-Means Clustering

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# OVERVIEW
# ============================================================

if menu == "🏠 Overview":

    st.markdown(
        '<div class="page-title">'
        'Customer Intelligence Dashboard'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="page-subtitle">'
        'Predict customer value, discover customer groups, '
        'and manage analytics data from one platform.'
        '</div>',

        unsafe_allow_html=True
    )


    # Hero section

    st.markdown("""
    <div class="hero">

        <div class="hero-badge">
            ✦ AI-POWERED CUSTOMER ANALYTICS
        </div>

        <h1>
            Understand your customers better.
        </h1>

        <p>
            CLV Intelligence combines Machine Learning
            and customer behaviour analytics to estimate
            Customer Lifetime Value and segment customers
            using Recency, Frequency and Monetary behaviour.
        </p>

    </div>
    """, unsafe_allow_html=True)


    # ========================================================
    # KPI CARDS
    # ========================================================

    saved_count = 0

    if os.path.exists(
        "user_inputs.xlsx"
    ):

        try:

            saved_count = len(
                pd.read_excel(
                    "user_inputs.xlsx"
                )
            )

        except Exception:

            saved_count = 0


    cols = st.columns(4)


    kpis = [

        (
            "🤖",
            "ML MODEL",
            "GBR",
            "Gradient Boosting Regressor"
        ),

        (
            "👥",
            "SEGMENTS",
            "3",
            "K-Means customer groups"
        ),

        (
            "📌",
            "FEATURES",
            "3",
            "Recency • Frequency • Monetary"
        ),

        (
            "💾",
            "PREDICTIONS",
            str(saved_count),
            "Stored prediction records"
        )
    ]


    for col, data in zip(
        cols,
        kpis
    ):

        icon, label, value, note = data

        with col:

            st.markdown(
                f"""
                <div class="kpi">

                    <div class="kpi-icon">
                        {icon}
                    </div>

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                    <div class="kpi-note">
                        {note}
                    </div>

                </div>
                """,

                unsafe_allow_html=True
            )


    # ========================================================
    # MODULES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '⚡ Platform Modules'
        '</div>',

        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    modules = [

        (
            "🔮",
            "CLV Predictor",
            "Estimate the future value of a customer "
            "using a trained Machine Learning model."
        ),

        (
            "👥",
            "Segmentation",
            "Group customers into meaningful clusters "
            "using K-Means clustering."
        ),

        (
            "🔐",
            "Admin Center",
            "Securely review prediction records and "
            "download stored Excel data."
        )
    ]


    for col, data in zip(
        [c1, c2, c3],
        modules
    ):

        icon, title, desc = data

        with col:

            st.markdown(
                f"""
                <div class="card"
                     style="height:170px;">

                    <div style="
                        font-size:30px;
                    ">
                        {icon}
                    </div>

                    <h3 style="
                        margin:8px 0;
                        color:#0f172a;
                    ">
                        {title}
                    </h3>

                    <p style="
                        color:#64748b;
                        font-size:13px;
                        line-height:1.6;
                    ">
                        {desc}
                    </p>

                </div>
                """,

                unsafe_allow_html=True
            )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🧠 How It Works'
        '</div>',

        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    steps = [

        (
            "01",
            "Input",
            "Enter customer behaviour"
        ),

        (
            "02",
            "Predict",
            "ML estimates CLV"
        ),

        (
            "03",
            "Segment",
            "K-Means groups customers"
        ),

        (
            "04",
            "Analyze",
            "Admin reviews stored data"
        )
    ]


    for col, data in zip(
        [c1, c2, c3, c4],
        steps
    ):

        num, title, desc = data

        with col:

            st.markdown(
                f"""
                <div class="card"
                     style="min-height:135px;">

                    <div style="
                        color:#6366f1;
                        font-weight:800;
                        font-size:12px;
                    ">
                        STEP {num}
                    </div>

                    <h4 style="
                        margin:8px 0;
                        color:#0f172a;
                    ">
                        {title}
                    </h4>

                    <p style="
                        font-size:12px;
                        color:#64748b;
                    ">
                        {desc}
                    </p>

                </div>
                """,

                unsafe_allow_html=True
            )


# ============================================================
# CLV PREDICTOR
# ============================================================

elif menu == "🔮 CLV Predictor":

    st.markdown(
        '<div class="page-title">'
        '🔮 CLV Predictor'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="page-subtitle">'
        'Estimate Customer Lifetime Value using customer '
        'purchasing behaviour.'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-title" '
        'style="margin-top:0;">'
        'Customer Behaviour'
        '</div>',

        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        recency = st.number_input(

            "📅 Recency (Days)",

            min_value=0,

            max_value=365,

            value=30,

            help=
            "Number of days since the customer's last purchase."
        )


    with c2:

        frequency = st.number_input(

            "🛒 Purchase Frequency",

            min_value=1,

            max_value=100,

            value=5,

            help=
            "Number of purchases made by the customer."
        )


    with c3:

        monetary = st.number_input(

            "💰 Monetary Value",

            min_value=0.0,

            max_value=10000.0,

            value=500.0,

            step=50.0,

            help=
            "Total or representative monetary value of purchases."
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    predict = st.button(

        "🚀 Predict Customer Lifetime Value",

        use_container_width=True,

        type="primary"
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict:

        model = train_clv_model()


        user_data = pd.DataFrame({

            "Recency": [recency],

            "Frequency": [frequency],

            "Monetary": [monetary]

        })


        prediction = float(

            model.predict(
                user_data
            )[0]
        )


        user_data[
            "Predicted_CLV"
        ] = prediction


        save_prediction(
            user_data
        )


        st.session_state.last_prediction = prediction


        st.markdown(

            f"""
            <div class="prediction">

                <div class="prediction-label">
                    ESTIMATED CUSTOMER LIFETIME VALUE
                </div>

                <div class="prediction-value">
                    ${prediction:,.2f}
                </div>

                <div class="prediction-caption">
                    Generated using Gradient Boosting Regression
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "📅 Recency",
                f"{recency} days"
            )


        with c2:

            st.metric(
                "🛒 Frequency",
                f"{frequency} purchases"
            )


        with c3:

            st.metric(
                "💰 Monetary",
                f"${monetary:,.2f}"
            )


        st.success(
            "✅ Prediction generated and stored successfully."
        )


    elif (
        st.session_state.last_prediction
        is not None
    ):

        st.info(
            f"Last prediction: "
            f"${st.session_state.last_prediction:,.2f}"
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '💡 CLV Interpretation'
        '</div>',

        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    interpretations = [

        (
            "📅 Recency",

            "Lower recency generally indicates "
            "a more recently active customer."
        ),

        (
            "🛒 Frequency",

            "Higher purchase frequency can indicate "
            "stronger customer engagement."
        ),

        (
            "💰 Monetary",

            "Higher spending contributes strongly "
            "to customer value."
        )
    ]


    for col, data in zip(
        [c1, c2, c3],
        interpretations
    ):

        title, desc = data

        with col:

            st.markdown(
                f"""
                <div class="card"
                     style="min-height:130px;">

                    <h4 style="
                        margin:0 0 8px;
                        color:#0f172a;
                    ">
                        {title}
                    </h4>

                    <p style="
                        margin:0;
                        color:#64748b;
                        font-size:13px;
                        line-height:1.6;
                    ">
                        {desc}
                    </p>

                </div>
                """,

                unsafe_allow_html=True
            )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif menu == "👥 Customer Segmentation":

    st.markdown(
        '<div class="page-title">'
        '👥 Customer Segmentation'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="page-subtitle">'
        'Discover customer groups using K-Means clustering.'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(

        "📁 Upload Customer CSV Dataset",

        type=["csv"],

        help=
        "Upload a CSV containing customer behaviour columns."
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )


            if df.empty:

                st.warning(
                    "The uploaded CSV is empty."
                )

                st.stop()


            # ==================================================
            # DATASET OVERVIEW
            # ==================================================

            st.markdown(
                '<div class="section-title">'
                '📋 Dataset Overview'
                '</div>',

                unsafe_allow_html=True
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "👥 Customers",
                    len(df)
                )


            with c2:

                st.metric(
                    "📊 Columns",
                    len(df.columns)
                )


            with c3:

                st.metric(
                    "🧹 Missing Values",
                    int(
                        df.isna()
                        .sum()
                        .sum()
                    )
                )


            st.dataframe(

                df.head(10),

                use_container_width=True,

                hide_index=True
            )


            # ==================================================
            # FEATURE SELECTION
            # ==================================================

            st.markdown(
                '<div class="section-title">'
                '⚙️ Select Features'
                '</div>',

                unsafe_allow_html=True
            )


            numeric_cols = (
                df
                .select_dtypes(
                    include="number"
                )
                .columns
                .tolist()
            )


            if len(numeric_cols) < 3:

                st.error(
                    "Please upload a CSV containing "
                    "at least 3 numeric columns."
                )

                st.stop()


            c1, c2, c3 = st.columns(3)


            with c1:

                r = st.selectbox(

                    "📅 Recency Column",

                    numeric_cols,

                    index=0
                )


            with c2:

                f = st.selectbox(

                    "🛒 Frequency Column",

                    numeric_cols,

                    index=min(
                        1,
                        len(numeric_cols) - 1
                    )
                )


            with c3:

                m = st.selectbox(

                    "💰 Monetary Column",

                    numeric_cols,

                    index=min(
                        2,
                        len(numeric_cols) - 1
                    )
                )


            if len(
                {
                    r,
                    f,
                    m
                }
            ) < 3:

                st.warning(
                    "Please select three different columns."
                )

                st.stop()


            # ==================================================
            # RUN SEGMENTATION
            # ==================================================

            if st.button(

                "🚀 Run Customer Segmentation",

                use_container_width=True,

                type="primary"
            ):

                work = df[
                    [
                        r,
                        f,
                        m
                    ]
                ].copy()


                work = work.dropna()


                if len(work) < 3:

                    st.error(
                        "At least 3 valid customer rows are required."
                    )

                    st.stop()


                scaler = StandardScaler()


                X_scaled = scaler.fit_transform(
                    work
                )


                kmeans = KMeans(

                    n_clusters=3,

                    random_state=42,

                    n_init=10
                )


                labels = kmeans.fit_predict(
                    X_scaled
                )


                result = df.loc[
                    work.index
                ].copy()


                result[
                    "Cluster"
                ] = labels


                # ==================================================
                # CLUSTER PROFILES
                # ==================================================

                profiles = (

                    result
                    .groupby("Cluster")[
                        [
                            r,
                            f,
                            m
                        ]
                    ]
                    .mean()
                    .reset_index()
                )


                overall_monetary = (
                    profiles[m].mean()
                )


                def label_cluster(row):

                    if (
                        row[m]
                        >= overall_monetary * 1.15
                    ):

                        return "High Value"

                    elif (
                        row[m]
                        <= overall_monetary * 0.85
                    ):

                        return "Low Value"

                    return "Regular"


                profiles[
                    "Segment"
                ] = profiles.apply(
                    label_cluster,
                    axis=1
                )


                result[
                    "Segment"
                ] = result[
                    "Cluster"
                ].map(
                    profiles
                    .set_index(
                        "Cluster"
                    )["Segment"]
                )


                st.success(
                    "✅ Customer segmentation completed successfully."
                )


                # ==================================================
                # RESULT METRICS
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    '📈 Segmentation Results'
                    '</div>',

                    unsafe_allow_html=True
                )


                c1, c2, c3, c4 = st.columns(4)


                with c1:

                    st.metric(
                        "👥 Customers",
                        len(result)
                    )


                with c2:

                    st.metric(
                        "🎯 Clusters",
                        result[
                            "Cluster"
                        ].nunique()
                    )


                with c3:

                    st.metric(
                        "💰 Avg Monetary",
                        f"${result[m].mean():,.2f}"
                    )


                with c4:

                    st.metric(
                        "🛒 Avg Frequency",
                        f"{result[f].mean():,.1f}"
                    )


                # ==================================================
                # CUSTOMER GROUP CARDS
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    '🏷️ Customer Groups'
                    '</div>',

                    unsafe_allow_html=True
                )


                high = len(
                    result[
                        result[
                            "Segment"
                        ]
                        == "High Value"
                    ]
                )


                regular = len(
                    result[
                        result[
                            "Segment"
                        ]
                        == "Regular"
                    ]
                )


                low = len(
                    result[
                        result[
                            "Segment"
                        ]
                        == "Low Value"
                    ]
                )


                c1, c2, c3 = st.columns(3)


                cards = [

                    (
                        "💎",
                        "High Value",
                        high,
                        "Customers with comparatively "
                        "higher monetary value."
                    ),

                    (
                        "⭐",
                        "Regular",
                        regular,
                        "Customers with balanced "
                        "purchasing behaviour."
                    ),

                    (
                        "📌",
                        "Low Value",
                        low,
                        "Customers with comparatively "
                        "lower monetary value."
                    )
                ]


                for col, data in zip(
                    [c1, c2, c3],
                    cards
                ):

                    icon, title, count, desc = data


                    with col:

                        st.markdown(
                            f"""
                            <div class="segment">

                                <div style="
                                    font-size:27px;
                                ">
                                    {icon}
                                </div>

                                <h3>
                                    {title}
                                </h3>

                                <div style="
                                    font-size:26px;
                                    font-weight:800;
                                    color:#4f46e5;
                                ">
                                    {count}
                                </div>

                                <p>
                                    {desc}
                                </p>

                            </div>
                            """,

                            unsafe_allow_html=True
                        )


                # ==================================================
                # DISTRIBUTION CHART
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    '📊 Customer Distribution'
                    '</div>',

                    unsafe_allow_html=True
                )


                counts = (
                    result[
                        "Segment"
                    ]
                    .value_counts()
                )


                fig1, ax1 = plt.subplots(
                    figsize=(8, 4.5)
                )


                counts.plot(
                    kind="bar",
                    ax=ax1
                )


                ax1.set_xlabel(
                    "Customer Segment"
                )


                ax1.set_ylabel(
                    "Number of Customers"
                )


                ax1.set_title(
                    "Customer Segment Distribution",
                    fontweight="bold"
                )


                ax1.tick_params(
                    axis="x",
                    rotation=0
                )


                ax1.grid(
                    axis="y",
                    alpha=0.2
                )


                st.pyplot(
                    fig1
                )


                # ==================================================
                # SCATTER CHART
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    '🔎 Behaviour Visualization'
                    '</div>',

                    unsafe_allow_html=True
                )


                fig2, ax2 = plt.subplots(
                    figsize=(10, 5)
                )


                ax2.scatter(

                    result[r],

                    result[m],

                    c=result[
                        "Cluster"
                    ],

                    s=75,

                    alpha=0.8
                )


                ax2.set_xlabel(
                    r
                )


                ax2.set_ylabel(
                    m
                )


                ax2.set_title(

                    f"{r} vs {m} — "
                    f"Customer Clusters",

                    fontweight="bold"
                )


                ax2.grid(
                    alpha=0.2
                )


                st.pyplot(
                    fig2
                )


                # ==================================================
                # SEGMENTED DATA
                # ==================================================

                st.markdown(
                    '<div class="section-title">'
                    '📄 Segmented Customer Data'
                    '</div>',

                    unsafe_allow_html=True
                )


                st.dataframe(

                    result,

                    use_container_width=True,

                    hide_index=True
                )


                csv_data = (
                    result
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )


                st.download_button(

                    "📥 Download Segmented CSV",

                    data=csv_data,

                    file_name=
                    "segmented_customers.csv",

                    mime="text/csv",

                    use_container_width=True
                )


        except Exception as e:

            st.error(
                f"Unable to process the dataset: {e}"
            )


# ============================================================
# ADMIN CENTER
# ============================================================

elif menu == "🔐 Admin Center":

    st.markdown(
        '<div class="page-title">'
        '🔐 Admin Center'
        '</div>',

        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="page-subtitle">'
        'Securely manage stored CLV prediction records.'
        '</div>',

        unsafe_allow_html=True
    )


    # ========================================================
    # LOGGED IN ADMIN
    # ========================================================

    if st.session_state.admin_logged_in:

        c1, c2 = st.columns(
            [
                4,
                1
            ]
        )


        with c1:

            st.success(
                "🟢 Administrator session active"
            )


        with c2:

            if st.button(
                "Logout",
                use_container_width=True
            ):

                st.session_state.admin_logged_in = False

                st.rerun()


        # ====================================================
        # SERVER FILES
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📂 Server Files'
            '</div>',

            unsafe_allow_html=True
        )


        files = os.listdir()


        st.dataframe(

            pd.DataFrame(
                {
                    "Files": files
                }
            ),

            use_container_width=True,

            hide_index=True
        )


        # ====================================================
        # STORED DATA
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📄 Stored Prediction Records'
            '</div>',

            unsafe_allow_html=True
        )


        try:

            stored = pd.read_excel(
                "user_inputs.xlsx"
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "👥 Records",
                    len(stored)
                )


            with c2:

                if (
                    "Predicted_CLV"
                    in stored.columns
                ):

                    avg = (
                        stored[
                            "Predicted_CLV"
                        ].mean()
                    )

                else:

                    avg = 0


                st.metric(
                    "💰 Average CLV",
                    f"${avg:,.2f}"
                )


            with c3:

                if (
                    "Predicted_CLV"
                    in stored.columns
                ):

                    max_clv = (
                        stored[
                            "Predicted_CLV"
                        ].max()
                    )

                else:

                    max_clv = 0


                st.metric(
                    "🏆 Highest CLV",
                    f"${max_clv:,.2f}"
                )


            st.dataframe(

                stored,

                use_container_width=True,

                hide_index=True
            )


            with open(
                "user_inputs.xlsx",
                "rb"
            ) as file:

                st.download_button(

                    "📥 Download Excel Data",

                    data=file,

                    file_name=
                    "user_inputs.xlsx",

                    mime=
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                    use_container_width=True
                )


        except Exception:

            st.info(
                "ℹ️ No prediction data has been stored yet."
            )


    # ========================================================
    # ADMIN LOGIN
    # ========================================================

    else:

        # ====================================================
        # BLOCKED
        # ====================================================

        if st.session_state.blocked:

            st.error(
                "🚫 Access blocked after "
                "3 incorrect attempts."
            )


            email = (
                "atharavshende999@gmail.com"
            )


            subject = (
                "Access Request for CLV App"
            )


            gmail_link = (

                "https://mail.google.com/mail/"
                "?view=cm"
                "&fs=1"
                f"&to={email}"
                f"&su={subject}"
            )


            st.markdown(
                """
                <div class="card">

                    <h3 style="
                        color:#0f172a;
                    ">
                        📩 Contact Administrator
                    </h3>

                    <p style="
                        color:#64748b;
                    ">
                        Access is temporarily blocked.
                        Contact the administrator
                        to request access.
                    </p>

                </div>
                """,

                unsafe_allow_html=True
            )


            st.markdown(

                f"""
                <a href="{gmail_link}"
                   target="_blank"
                   style="text-decoration:none;">

                    <div style="
                        display:inline-block;
                        margin-top:15px;
                        padding:13px 24px;
                        border-radius:11px;
                        color:white;
                        font-weight:700;
                        background:
                        linear-gradient(
                            135deg,
                            #ef4444,
                            #dc2626
                        );
                    ">

                        📧 Contact Admin

                    </div>

                </a>
                """,

                unsafe_allow_html=True
            )


            st.stop()


        # ====================================================
        # LOGIN UI
        # ====================================================

        st.markdown(
            """
            <div class="card">

                <div style="
                    font-size:42px;
                    text-align:center;
                ">
                    🔐
                </div>

                <h2 style="
                    text-align:center;
                    color:#0f172a;
                    margin:10px 0;
                ">
                    Administrator Login
                </h2>

                <p style="
                    text-align:center;
                    color:#64748b;
                    font-size:13px;
                ">
                    Enter the administrator password
                    to access stored customer data.
                </p>

            </div>
            """,

            unsafe_allow_html=True
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(
            [
                1,
                2,
                1
            ]
        )


        with c2:

            password = st.text_input(

                "🔑 Admin Password",

                type="password",

                placeholder=
                "Enter password"
            )


            login = st.button(

                "🔓 Sign In",

                use_container_width=True,

                type="primary"
            )


            st.caption(

                f"Security: "
                f"{3 - st.session_state.attempts} "
                f"attempts remaining"
            )


            # =================================================
            # LOGIN VALIDATION
            # =================================================

            if login:

                # ------------------------------------------------
                # DEMO PASSWORD
                # For production use st.secrets instead.
                # ------------------------------------------------

                if password.strip() == "admin123":

                    st.session_state.attempts = 0

                    st.session_state.admin_logged_in = True

                    st.success(
                        "✅ Access granted."
                    )

                    st.rerun()

                else:

                    st.session_state.attempts += 1

                    remaining = (
                        3
                        -
                        st.session_state.attempts
                    )


                    if remaining > 0:

                        st.error(

                            f"❌ Incorrect password. "
                            f"{remaining} attempt(s) remaining."
                        )

                    else:

                        st.session_state.blocked = True

                        st.rerun()
