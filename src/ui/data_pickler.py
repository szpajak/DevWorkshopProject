import shap
import transformers
import pickle
import numpy as np
import pandas as pd
import os
from typing import List
from transformers import AutoTokenizer

from common import (
    get_raw_data,
    get_tokenizer,
    load_model_from_file,
    preprocess,
)

# CD012179,   CD010502,    CD11431

topic = "CD012010" # Example topic, consider making this configurable

model = load_model_from_file(topic)
tokenizer = get_tokenizer()

def load_data(topic):
    train_df, test_df = get_raw_data()
    return preprocess(topic=topic, df=train_df)

_train_dataset, test_dataset, _train_loader, _test_loader = load_data(topic)

sent_analyzer = transformers.pipeline(
    "sentiment-analysis",
    tokenizer=tokenizer,
    model=model.bert,
    top_k=None # Get scores for all classes
)

# Explicitly create a Text masker using a regex for word-level tokenization
masker = shap.maskers.Text(r"\b\w+\b|[^\w\s]")
explainer = shap.Explainer(sent_analyzer, masker=masker, output_names=["Irrelevant", "Relevant"])

label_map = {
    "LABEL_0": "Irrelevant",
    "LABEL_1": "Relevant"
}

articles_to_process = []
# num_items_to_process = min(2, len(test_dataset.X))
start = 10
stop = 25

for i in range(start, stop):
    row = test_dataset.X.iloc[i]
    text_title = str(row.title) if hasattr(row, 'title') and pd.notna(row.title) else ""
    text_abstract = str(row.abstract) if hasattr(row, 'abstract') and pd.notna(row.abstract) else ""

    if not text_title and not text_abstract:
        print(f"Skipping row {i} due to empty title and abstract.")
        continue

    # Combine title and abstract for prediction, but keep them separate for SHAP
    text_input_for_prediction = f"{text_title} {text_abstract}".strip()
    if not text_input_for_prediction:
        print(f"Skipping row {i} due to empty combined text for title: {text_title[:30]}")
        continue

    print(f"\nProcessing article {i+1}/{stop} in data_pickler: {text_title[:60]}...")

    shap_explanations_title = None
    shap_explanations_abstract = None
    try:
        # Generate SHAP explanations for title and abstract separately
        # Note: We process them in a batch for efficiency
        texts_to_explain = []
        if text_title:
            texts_to_explain.append(text_title)
        if text_abstract:
            texts_to_explain.append(text_abstract)

        if texts_to_explain:
            shap_explanations_list = explainer(texts_to_explain)
            
            # Assign explanations back
            title_idx = 0 if text_title else -1
            abstract_idx = 0 if not text_title and text_abstract else 1 if text_title and text_abstract else -1

            if title_idx != -1:
                shap_explanations_title = shap_explanations_list[title_idx]
            if abstract_idx != -1:
                shap_explanations_abstract = shap_explanations_list[abstract_idx]

    except Exception as e:
        print(f"  Error during SHAP explanation for article {text_title[:60]}: {e}")
        # Continue with prediction even if SHAP fails
    
    prediction = "Error"
    try:
        # Use the combined text for the final verdict prediction
        prediction_result = sent_analyzer(text_input_for_prediction)
        if isinstance(prediction_result, list) and len(prediction_result) > 0:
            top_pred_dict = prediction_result[0]
            if isinstance(top_pred_dict, list) and len(top_pred_dict) > 0: top_pred_dict = top_pred_dict[0]
            if isinstance(top_pred_dict, dict):
                prediction_label = top_pred_dict.get("label")
                prediction = label_map.get(prediction_label, prediction_label)
    except Exception as e:
        print(f"  Error during sentiment analysis for prediction for article {text_title[:60]}: {e}")

    # Debugging output
    if shap_explanations_title:
        print(f"  data_pickler: Saving SHAP explanation for TITLE. Tokens: {len(getattr(shap_explanations_title, 'data', []))}")
    if shap_explanations_abstract:
        print(f"  data_pickler: Saving SHAP explanation for ABSTRACT. Tokens: {len(getattr(shap_explanations_abstract, 'data', []))}")

    articles_to_process.append({
        "title": text_title,
        "abstract": text_abstract,
        "shap_title_values": shap_explanations_title,
        "shap_abstract_values": shap_explanations_abstract,
        "verdict": prediction
    })


base_filename = f"shap/shaps_{topic}.pkl"
os.makedirs(os.path.dirname(base_filename), exist_ok=True)

output_filename = base_filename
counter = 1
while os.path.exists(output_filename):
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_{counter}{ext}"
    counter += 1


with open(output_filename, "wb") as f:
    pickle.dump(articles_to_process, f)

print(f"\nSuccessfully processed {len(articles_to_process)} articles and saved to {output_filename}")