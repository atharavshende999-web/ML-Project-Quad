import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEXA CLV | Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "user_inputs.xlsx"


# ============================================================
# PROFESSIONAL WEBSITE THEME
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: #0b1020;
    border-right: 1px solid #20283d;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 4px;
}

/* ---------- TOP BRAND ---------- */

.brand {
    font-size: 25px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.8px;
}

.brand span {
    color: #7c3aed;
}

.brand-sub {
    color: #8992a8;
    font-size: 11px;
    letter-spacing: 1.6px;
    margin-top: -5px;
}

/* ---------- HERO ---------- */

.hero {
    background:
        radial-gradient(
            circle at 85% 15%,
            rgba(124,58,237,0.32),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #0b1020,
            #111936 55%,
            #172554
        );

    border-radius: 26px;
    padding: 46px;
    margin-bottom: 28px;
    color: white;
    border: 1px solid #263354;
    box-shadow: 0 18px 50px rgba(15,23,42,0.18);
}

.hero-tag {
    display: inline-block;
    background: rgba(124,58,237,0.18);
    border: 1px solid rgba(167,139,250,0.35);
    color: #c4b5fd;
    border-radius: 30px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .5px;
}

.hero h1 {
    font-size: 48px;
    line-height: 1.08;
    margin: 18px 0 12px 0;
    letter-spacing: -2px;
    color: white;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
    max-width: 700px;
    line-height: 1.7;
}

/* ---------- SECTION ---------- */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #111827;
    margin: 24px 0 6px 0;
}

.section-sub {
    color: #64748b;
    margin-bottom: 18px;
}

/* ---------- CARDS ---------- */

.card {
    background: white;
    border: 1px solid #e7eaf0;
    border-radius: 18px;
    padding: 23px;
    min-height: 180px;
    box-shadow: 0 8px 25px rgba(15,23,42,.045);
}

.card-icon {
    font-size: 30px;
}

.card-title {
    font-size: 18px;
    font-weight: 750;
    color: #111827;
    margin-top: 10px;
}

.card-text {
    color: #64748b;
    line-height: 1.55;
    font-size: 14px;
}

/* ---------- METRICS ---------- */

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e7eaf0;
    border-radius: 16px;
    padding: 17px;
    box-shadow: 0 6px 20px rgba(15,23,42,.04);
}

/* ---------- INPUTS ---------- */

div[data-baseweb="input"] {
    border-radius: 10px;
}

div[data-baseweb="select"] > div {
    border-radius: 10px;
}

/* ---------- BUTTONS ---------- */

.stButton button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 44px;
}

/* ---------- INFO BOX ---------- */

div[data-testid="stAlert"] {
    border-radius: 13px;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 35px 0 10px 0;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "admin" not in st.session_state:
    st.session_state.admin = False

if "attempts" not in st.session_state:
    st.session_state.attempts = 0

if "blocked" not in st.session_state:
    st.session_state.blocked = False


# ============================================================
# ML MODEL
# ============================================================

@st.cache_resource
def build_model():

    df = pd.DataFrame({

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
    })

    X = df[
        ["Recency", "Frequency", "Monetary"]
    ]

    y = df["CLV"]

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )

    model.fit(X, y)

    return model


# ============================================================
# DATA STORAGE
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        return None

    try:
        return pd.read_excel(DATA_FILE)
    except:
        return None


def save_data(row):

    try:

        if os.path.exists(DATA_FILE):

            old = pd.read_excel(DATA_FILE)

            new = pd.concat(
                [old, row],
                ignore_index=True
            )

        else:

            new = row

        new.to_excel(
            DATA_FILE,
            index=False
        )

        return True

    except Exception as e:

        st.error(
            f"Storage error: {e}"
        )

        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            NEXA<span>CLV</span>
        </div>

        <div class="brand-sub">
            CUSTOMER INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "⌂ Overview",
            "◈ CLV Prediction",
            "◉ Customer Segmentation",
            "▦ Analytics",
            "⚙ Administration"
        ]
    )

    st.divider()

    st.caption("PLATFORM STATUS")

    st.success(
        "● ML ENGINE ONLINE"
    )

    st.caption(
        "Gradient Boosting"
    )

    st.caption(
        "K-Means Clustering"
    )

    st.divider()

    st.caption(
        "NEXA CLV v2.0"
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "⌂ Overview":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-tag">
                ✦ MACHINE LEARNING CUSTOMER INTELLIGENCE
            </div>

            <h1>
                Know which customers<br>
                drive your future.
            </h1>

            <p>
                NEXA CLV transforms customer behaviour into
                actionable intelligence using machine learning.
                Predict lifetime value, identify customer segments,
                and understand the customers that matter most.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    data = load_data()

    customers = 0
    average = 0
    maximum = 0

    if data is not None and not data.empty:

        customers = len(data)

        if "Predicted_CLV" in data.columns:

            average = data[
                "Predicted_CLV"
            ].mean()

            maximum = data[
                "Predicted_CLV"
            ].max()

    st.markdown(
        '<div class="section-title">Live Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'A real-time view of your customer value ecosystem.'
        '</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Customers Analysed",
        customers
    )

    b.metric(
        "Average CLV",
        f"${average:,.0f}"
    )

    c.metric(
        "Highest CLV",
        f"${maximum:,.0f}"
    )

    d.metric(
        "ML Engine",
        "ONLINE"
    )

    st.write("")

    # ========================================================
    # FEATURE CARDS
    # ========================================================

    f1, f2, f3 = st.columns(3)

    with f1:

        st.markdown(
            """
            <div class="card">

                <div class="card-icon">🧠</div>

                <div class="card-title">
                    Machine Learning Prediction
                </div>

                <div class="card-text">
                    Gradient Boosting analyses customer
                    behaviour and estimates future
                    Customer Lifetime Value.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:

        st.markdown(
            """
            <div class="card">

                <div class="card-icon">👥</div>

                <div class="card-title">
                    Behaviour Segmentation
                </div>

                <div class="card-text">
                    K-Means clustering separates customers
                    into meaningful behavioural groups.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:

        st.markdown(
            """
            <div class="card">

                <div class="card-icon">📊</div>

                <div class="card-title">
                    Decision Intelligence
                </div>

                <div class="card-text">
                    Convert raw customer data into
                    visual insights that support
                    smarter business decisions.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ========================================================
    # ML PIPELINE
    # ========================================================

    st.markdown(
        '<div class="section-title">How the Intelligence Engine Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'From customer behaviour to business intelligence.'
        '</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.info(
            "**01  DATA**\n\n"
            "Customer transaction behaviour"
        )

    with p2:
        st.info(
            "**02  RFM**\n\n"
            "Recency • Frequency • Monetary"
        )

    with p3:
        st.info(
            "**03  MODEL**\n\n"
            "Gradient Boosting prediction"
        )

    with p4:
        st.info(
            "**04  INSIGHT**\n\n"
            "Value & segmentation intelligence"
        )

    # ========================================================
    # PREVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">Customer Value Preview</div>',
        unsafe_allow_html=True
    )

    if data is not None and "Predicted_CLV" in data.columns:

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            data["Predicted_CLV"],
            linewidth=2,
            marker="o"
        )

        ax.set_title(
            "Predicted Customer Lifetime Value"
        )

        ax.set_xlabel(
            "Customer"
        )

        ax.set_ylabel(
            "CLV"
        )

        ax.grid(
            alpha=.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Your CLV intelligence chart will appear here "
            "after the first prediction."
        )


# ============================================================
# CLV PREDICTION
# ============================================================

elif page == "◈ CLV Prediction":

    st.markdown(
        '<div class="section-title">Customer Lifetime Value Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Enter customer behaviour and let the ML engine estimate future value.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 1.2]
    )

    with left:

        st.markdown(
            """
            <div class="card">
            <b>Customer Behaviour Input</b>
            <br><br>
            Provide the customer's RFM information.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        recency = st.number_input(
            "Recency",
            min_value=0,
            max_value=365,
            value=30,
            help="Days since the customer's last purchase."
        )

        frequency = st.number_input(
            "Frequency",
            min_value=1,
            max_value=100,
            value=5,
            help="Number of purchases."
        )

        monetary = st.number_input(
            "Monetary Value",
            min_value=0.0,
            max_value=100000.0,
            value=500.0,
            step=50.0,
            help="Total amount spent."
        )

        predict = st.button(
            "✦ RUN ML PREDICTION",
            type="primary",
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section-title">Customer Profile</div>',
            unsafe_allow_html=True
        )

        x1, x2, x3 = st.columns(3)

        x1.metric(
            "RECENCY",
            f"{recency} days"
        )

        x2.metric(
            "FREQUENCY",
            frequency
        )

        x3.metric(
            "MONETARY",
            f"${monetary:,.0f}"
        )

        st.write("")

        st.info(
            """
            **Model Input**

            These three behavioural signals are processed
            by the Gradient Boosting model to estimate
            Customer Lifetime Value.
            """
        )

    if predict:

        model = build_model()

        customer = pd.DataFrame({

            "Recency": [recency],

            "Frequency": [frequency],

            "Monetary": [monetary]

        })

        prediction = float(
            model.predict(customer)[0]
        )

        customer[
            "Predicted_CLV"
        ] = prediction

        save_data(
            customer
        )

        st.session_state.prediction = prediction

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction

        st.divider()

        st.markdown(
            '<div class="section-title">ML Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction >= 2000:

            level = "HIGH VALUE"
            icon = "🟢"

        elif prediction >= 1000:

            level = "MEDIUM VALUE"
            icon = "🟡"

        else:

            level = "STANDARD VALUE"
            icon = "🔵"

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "PREDICTED CLV",
            f"${prediction:,.2f}"
        )

        r2.metric(
            "CUSTOMER VALUE",
            f"{icon} {level}"
        )

        r3.metric(
            "MODEL",
            "Gradient Boosting"
        )

        st.success(
            "✓ Prediction completed successfully."
        )

        st.markdown(
            '<div class="section-title">RFM Behaviour Analysis</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.bar(
            ["Recency", "Frequency", "Monetary"],
            [
                recency,
                frequency,
                monetary
            ]
        )

        ax.set_title(
            "Customer Behaviour Profile"
        )

        ax.grid(
            axis="y",
            alpha=.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


# ============================================================
# SEGMENTATION
# ============================================================

elif page == "◉ Customer Segmentation":

    st.markdown(
        '<div class="section-title">Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Discover behavioural groups using unsupervised machine learning.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Upload a CSV containing customer Recency, Frequency and Monetary data."
    )

    uploaded = st.file_uploader(
        "Upload customer dataset",
        type=["csv"]
    )

    if uploaded:

        try:

            df = pd.read_csv(
                uploaded
            )

            st.success(
                f"Dataset loaded • {len(df):,} customer records"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.divider()

            st.markdown(
                '<div class="section-title">Feature Selection</div>',
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)

            recency_col = a.selectbox(
                "Recency",
                df.columns
            )

            frequency_col = b.selectbox(
                "Frequency",
                df.columns
            )

            monetary_col = c.selectbox(
                "Monetary",
                df.columns
            )

            cluster_count = st.slider(
                "Number of customer clusters",
                2,
                6,
                3
            )

            run = st.button(
                "✦ RUN CLUSTERING MODEL",
                type="primary",
                use_container_width=True
            )

            if run:

                X = df[
                    [
                        recency_col,
                        frequency_col,
                        monetary_col
                    ]
                ].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                valid = X.notna().all(
                    axis=1
                )

                X = X.loc[
                    valid
                ]

                result = df.loc[
                    valid
                ].copy()

                if len(X) < cluster_count:

                    st.error(
                        "There are not enough valid records "
                        "for the selected number of clusters."
                    )

                else:

                    scaler = StandardScaler()

                    scaled = scaler.fit_transform(
                        X
                    )

                    model = KMeans(
                        n_clusters=cluster_count,
                        random_state=42,
                        n_init=10
                    )

                    result["Cluster"] = model.fit_predict(
                        scaled
                    )

                    st.success(
                        "✓ K-Means segmentation completed."
                    )

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric(
                        "Customers",
                        len(result)
                    )

                    m2.metric(
                        "Clusters",
                        cluster_count
                    )

                    m3.metric(
                        "Avg Monetary",
                        f"${X[monetary_col].mean():,.0f}"
                    )

                    m4.metric(
                        "Avg Frequency",
                        f"{X[frequency_col].mean():.1f}"
                    )

                    st.divider()

                    st.markdown(
                        '<div class="section-title">Behavioural Map</div>',
                        unsafe_allow_html=True
                    )

                    fig, ax = plt.subplots(
                        figsize=(11, 5)
                    )

                    points = ax.scatter(
                        result[recency_col],
                        result[monetary_col],
                        c=result["Cluster"],
                        cmap="viridis",
                        s=80,
                        alpha=.85
                    )

                    ax.set_xlabel(
                        recency_col
                    )

                    ax.set_ylabel(
                        monetary_col
                    )

                    ax.set_title(
                        "Customer Behaviour Clusters"
                    )

                    ax.grid(
                        alpha=.2
                    )

                    plt.colorbar(
                        points,
                        ax=ax,
                        label="Cluster"
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    st.markdown(
                        '<div class="section-title">Segment Intelligence</div>',
                        unsafe_allow_html=True
                    )

                    summary = result.groupby(
                        "Cluster"
                    )[
                        [
                            recency_col,
                            frequency_col,
                            monetary_col
                        ]
                    ].mean().round(2)

                    st.dataframe(
                        summary,
                        use_container_width=True
                    )

                    st.download_button(
                        "↓ Download Segmented Customers",
                        result.to_csv(
                            index=False
                        ),
                        "customer_segments.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Dataset processing failed: {e}"
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "▦ Analytics":

    st.markdown(
        '<div class="section-title">Customer Intelligence Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Monitor the value distribution of your predicted customers.'
        '</div>',
        unsafe_allow_html=True
    )

    data = load_data()

    if data is None or data.empty:

        st.info(
            "No customer predictions are available yet."
        )

    elif "Predicted_CLV" not in data.columns:

        st.warning(
            "The stored data does not contain Predicted_CLV."
        )

    else:

        values = data[
            "Predicted_CLV"
        ]

        a, b, c, d = st.columns(4)

        a.metric(
            "Predictions",
            len(values)
        )

        b.metric(
            "Average CLV",
            f"${values.mean():,.0f}"
        )

        c.metric(
            "Maximum CLV",
            f"${values.max():,.0f}"
        )

        d.metric(
            "Minimum CLV",
            f"${values.min():,.0f}"
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.markdown(
                '<div class="section-title">CLV Distribution</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.hist(
                values,
                bins=10
            )

            ax.set_xlabel(
                "Predicted CLV"
            )

            ax.set_ylabel(
                "Customers"
            )

            ax.grid(
                alpha=.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        with right:

            st.markdown(
                '<div class="section-title">Value Trend</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.plot(
                values,
                marker="o"
            )

            ax.set_xlabel(
                "Customer"
            )

            ax.set_ylabel(
                "CLV"
            )

            ax.grid(
                alpha=.2
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Data</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            data,
            use_container_width=True
        )


# ============================================================
# ADMIN
# ============================================================

elif page == "⚙ Administration":

    st.markdown(
        '<div class="section-title">Administration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-sub">'
        'Secure management of stored prediction records.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.blocked:

        st.error(
            "Access blocked after 3 incorrect attempts."
        )

        st.link_button(
            "Contact Administrator",
            "https://mail.google.com/mail/?view=cm&fs=1&to=atharavshende999@gmail.com",
            use_container_width=True
        )

        st.stop()

    if not st.session_state.admin:

        st.info(
            "Administrator authentication required."
        )

        password = st.text_input(
            "Admin Password",
            type="password"
        )

        login = st.button(
            "SIGN IN",
            type="primary",
            use_container_width=True
        )

        if login:

            if password == "admin123":

                st.session_state.admin = True
                st.session_state.attempts = 0

                st.rerun()

            else:

                st.session_state.attempts += 1

                remaining = (
                    3 -
                    st.session_state.attempts
                )

                if remaining > 0:

                    st.error(
                        f"Incorrect password. "
                        f"{remaining} attempts remaining."
                    )

                else:

                    st.session_state.blocked = True

                    st.rerun()

    else:

        st.success(
            "Administrator authenticated."
        )

        if st.button(
            "LOG OUT",
            use_container_width=True
        ):

            st.session_state.admin = False

            st.rerun()

        data = load_data()

        if data is not None and not data.empty:

            st.divider()

            st.markdown(
                '<div class="section-title">Stored Intelligence</div>',
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)

            a.metric(
                "Records",
                len(data)
            )

            if "Predicted_CLV" in data.columns:

                b.metric(
                    "Average CLV",
                    f"${data['Predicted_CLV'].mean():,.0f}"
                )

                c.metric(
                    "Highest CLV",
                    f"${data['Predicted_CLV'].max():,.0f}"
                )

            st.dataframe(
                data,
                use_container_width=True
            )

            st.download_button(
                "↓ Download Prediction Records",
                data.to_csv(index=False),
                "clv_records.csv",
                "text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "No prediction records have been stored."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        NEXA CLV · MACHINE LEARNING CUSTOMER INTELLIGENCE
        <br>
        Gradient Boosting · K-Means · RFM Analytics
    </div>
    """,
    unsafe_allow_html=True
)
