import os

import pandas as pd
import requests
import streamlit as st

API_KEY = os.environ["DATAROBOT_API_TOKEN"]
DATAROBOT_KEY = "544ec55f-61bf-f6ee-0caf-15c7f919a45d"
DEPLOYMENT_ID = "644ffa6467e8bb38ba6370b4"
API_URL = f"https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{DEPLOYMENT_ID}/predictions"


HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "Authorization": "Bearer {}".format(API_KEY),
    "DataRobot-Key": DATAROBOT_KEY,
}

SMALL_FONT_STYLE = """
    <style>
    .small-font {
        font-size:14px;
        font-style: italic;
        color: #b1a7a6;
    }
    </style>
    """


@st.cache_data
def score_model(data):
    results = requests.post(
        API_URL, data=data.to_json(orient="records"), headers=HEADERS
    )
    data = results.json()["data"]
    return pd.Series([i["predictionValues"][0]["value"] for i in data])
