from datarobot_predict.scoring_code import ScoringCodeModel
import pandas as pd
from PIL import Image
import streamlit as st

import helpers as hf

PAGE_SIZE = 20


def load_model():
    return ScoringCodeModel("model.jar")


@st.cache_data
def load_data():
    return pd.read_csv("data/bleedout_train.csv").drop(columns="Bleedout")


@st.cache_data
def sample_data(df, records, seed=42):
    return df.sample(n=records, random_state=seed, replace=True).reset_index(drop=True)


st.set_page_config(page_title="Bleedout Predictor", layout="wide", page_icon="⚙️")

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
    "See the project here: https://app.datarobot.com/projects/643eab974c62a7e14b723c82/models"
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
notification_bar = st.empty()

scored_data_section = st.empty()
scored_data = pd.Series(name="Prediction")
scored_data_section.write("No data scored yet")

if pressed_score:
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
