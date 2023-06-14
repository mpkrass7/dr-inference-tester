from datarobot_predict.scoring_code import ScoringCodeModel
import pandas as pd
from PIL import Image
import streamlit as st
import time

import helpers as hf

PAGE_SIZE = 20


def load_model():
    return ScoringCodeModel("model.jar")


@st.cache_data
def load_data():
    return (
        pd.read_csv("data/bleedout_train.csv").drop(columns="Bleedout").sample(n=1000)
    )


@st.cache_data
def sample_data(df, records, seed=42):
    return df.sample(n=records, random_state=seed, replace=True).reset_index(drop=True)


st.set_page_config(page_title="Inference Tester", layout="wide", page_icon="⚙️")

st.markdown(
    hf.SMALL_FONT_STYLE,
    unsafe_allow_html=True,
)

data = load_data()
model = load_model()

if "data" not in st.session_state:
    st.session_state["data"] = data

st.title("DataRobot Batch/Real Time Inference Tester")
expander = st.expander("What is this?")
expander.markdown(
    """
    This application is designed to show how to score data from a DataRobot model using the
    Real Time API and the Batch Scoring exportable jar file.

    For context, our goal is to predict the probability of **bleedout** from a machine
    that creates coatings on layers of film. The normal process these machines go through
    to coat machine is shown in the image below:
    """
)
expander.image(Image.open("img/bleedout.jpg"), use_column_width=True)
expander.markdown(
    """
    See the project here: https://app.datarobot.com/projects/643eab974c62a7e14b723c82/models

    And the deployment here: https://app.datarobot.com/deployments/644ffa6467e8bb38ba6370b4/overview
    """
)

with st.sidebar.form(key="model_activation"):

    number_of_records = st.number_input(
        "Input the number of records to Score",
        min_value=1,
        max_value=1000000,
        value=1000,
    )
    pressed_scenario = st.form_submit_button("Generate Data")

    scoring_mode = st.radio("Scoring Mode", ("Batch", "Realtime API"))
    pressed_score = st.form_submit_button("Run")

if pressed_scenario:
    data = sample_data(data, number_of_records)
    st.session_state["data"] = data

st.dataframe(st.session_state["data"])

st.markdown("## Scoring", unsafe_allow_html=True)
expander = st.expander("How do I code this again?")
if scoring_mode == "Batch":
    expander.write(
        "Batch scoring with an exported model is easy to do and can be run in one line of code:"
    )
    expander.code(
        """
# import package
from datarobot_predict.scoring_code import ScoringCodeModel

# Make predictions
ScoringCodeModel("model.jar").predict(data)
"""
    )
elif scoring_mode == "Realtime API":
    expander.write(
        "Scoring with the Realtime API can be done using a simple POST request:"
    )
    expander.code(
        """
import requests
# Define Endpoint
API_URL = f"https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{DEPLOYMENT_ID}/predictions"

# Define Headers
HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "Authorization": "Bearer {}".format(API_KEY),
    "DataRobot-Key": DATAROBOT_KEY,
}

# Make Predictions
results = requests.post(
    API_URL, data=data.to_json(orient="records"), headers=HEADERS
)
    
"""
    )

notification_bar = st.empty()

scored_data_section = st.empty()
scoring_time_section = st.columns([1, 1, 2])
scored_data = pd.Series(name="Prediction", dtype="float64")
scored_data_section.write("No data scored yet")

if pressed_score:
    with st.spinner("Scoring..."):
        current_time = time.time()
        if scoring_mode == "Batch":
            scored_data_section.write(model.predict(st.session_state["data"]))
            notification_bar.info(f"Scored {len(st.session_state['data'])} records")
        else:
            for i in range(len(st.session_state["data"])):
                score_record = st.session_state["data"].iloc[i : i + 1, :]
                response = hf.score_model(score_record)
                scored_data = scored_data.append(response, ignore_index=True)
                notification_bar.info(
                    f"Scored {i + 1} records out of {len(st.session_state['data'])}"
                )
                scored_data_section.write(scored_data.rename("Prediction"))
    time_elapsed = round(time.time() - current_time, 3)
    average_time_per_record = round(time_elapsed / len(st.session_state["data"]), 5)
    scoring_time_section[0].markdown(
        f"**Total Inference Time (Seconds):** {time_elapsed}", unsafe_allow_html=True
    )
    scoring_time_section[1].markdown(
        f"**Average Inference Time per Record (Seconds):** {average_time_per_record}",
        unsafe_allow_html=True,
    )
