import streamlit as st
from streamlit_shap import st_shap
import shap
import transformers

from common import (
    get_raw_data,
    get_tokenizer,
    get_top_topics,
    load_model_from_file,
    preprocess,
)

model = load_model_from_file()
tokenizer = get_tokenizer()


def load_data():
    train_df, test_df = get_raw_data()
    top_topics = get_top_topics(train_df)
    return preprocess(topic=top_topics[0], df=train_df)


train_dataset, test_dataset, train_loader, test_loader = load_data()

st.set_page_config(layout="wide")
st.title("XAI for Citation Screening")

option = st.selectbox("Select topic title", tuple(test_dataset.X["title"]))

X = test_dataset.X[test_dataset.X["title"] == option].iloc[0]

X

sent_analyzer = transformers.pipeline(
    "sentiment-analysis",
    tokenizer=tokenizer,
    top_k=10,
    model=model.bert,
)
explainer = shap.Explainer(sent_analyzer, output_names=["Irrelavant", "Relevant"])
shap_values = explainer(X)

st_shap(shap.plots.text(shap_values), height=300)
