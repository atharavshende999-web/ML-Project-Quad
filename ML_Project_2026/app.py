# ============================================================
# AUTOMATIC DATASET LOADING
# NO UPLOAD BUTTON
# NO DATASET SELECTOR
# ============================================================

import os
import pandas as pd
import streamlit as st


@st.cache_data
def load_project_dataset():

    # Put your dataset filename here
    DATASET_PATH = "customer_clv.csv"

    if not os.path.exists(DATASET_PATH):
        st.error(
            f"Dataset '{DATASET_PATH}' was not found in the project folder."
        )
        st.stop()

    return pd.read_csv(DATASET_PATH)


df = load_project_dataset()
