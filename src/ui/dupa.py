import os
import pickle




# Load available topics
shap_dir = "shap"
topic_files = [f for f in os.listdir(shap_dir) if f.startswith("shaps_") and f.endswith(".pkl")]


topic_names = [f[6:-4] for f in topic_files]  # "shaps_<topic>.pkl" → "<topic>"

# Load SHAP values for selected topic
with open(os.path.join(shap_dir, f"shaps_CD012599.pkl"), "rb") as f:
    article_data = pickle.load(f)

print(article_data)