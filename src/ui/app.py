# streamlit_app.py
import streamlit as st
from streamlit_shap import st_shap
import shap
import pickle
import os

st.set_page_config(layout="wide")
st.title("📚 Explainable AI for Citation Screening")

# Load available topics
shap_dir = "shap"
topic_files = [f for f in os.listdir(shap_dir) if f.startswith("shaps_") and f.endswith(".pkl")]

if not topic_files:
    st.error("No SHAP files found.")
    st.stop()

topic_names = [f[6:-4] for f in topic_files]  # "shaps_<topic>.pkl" → "<topic>"
topic = st.selectbox("Select a topic:", topic_names)

# Load SHAP values for selected topic
with open(os.path.join(shap_dir, f"shaps_{topic}.pkl"), "rb") as f:
    article_data = pickle.load(f)

if not article_data:
    st.warning("No articles found for this topic.")
    st.stop()

article_titles = [d["title"] for d in article_data]
selected_title = st.selectbox("Select an article:", article_titles)

# Find selected article
selected_article = next(item for item in article_data if item["title"] == selected_title)

st.markdown(f"### 📰 {selected_article['title']}")
st.markdown(f"**Abstract:** {selected_article['abstract']}")
st.markdown(f"**Verdict:** :{'green' if selected_article['verdict']=='Relevant' else 'red'}[**{selected_article['verdict']}**]")

# Extract and display top 5 words
shap_vals = selected_article["shap_values"]
if hasattr(shap_vals, "data") and shap_vals.data is not None:
    tokens = shap_vals.data[0]
    scores = shap_vals.values[0]
    contributions = list(zip(tokens, scores))
    top_words = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
    top_words_str = ", ".join([f"`{word}`" for word, _ in top_words])
    st.markdown(f"**Top 5 influential words:** {top_words_str}")
else:
    st.warning("No SHAP tokens available to display top words.")

# Show SHAP visualization
st.markdown("**SHAP Explanation:**")
st_shap(shap.plots.text(shap_vals), height=300)
