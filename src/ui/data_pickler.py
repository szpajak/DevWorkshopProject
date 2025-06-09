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

topic = "CD012599" # Example topic, consider making this configurable

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

# Explicitly create a Text masker using the model's tokenizer
masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else "[MASK]")
explainer = shap.Explainer(sent_analyzer, masker=masker, output_names=["Irrelevant", "Relevant"])

label_map = {
    "LABEL_0": "Irrelevant",
    "LABEL_1": "Relevant"
}

articles_to_process = []
num_items_to_process = min(2, len(test_dataset.X))

for i in range(num_items_to_process):
    row = test_dataset.X.iloc[i]
    text_title = str(row.title) if hasattr(row, 'title') and pd.notna(row.title) else ""
    text_abstract = str(row.abstract) if hasattr(row, 'abstract') and pd.notna(row.abstract) else ""

    if not text_title and not text_abstract:
        print(f"Skipping row {i} due to empty title and abstract.")
        continue
    text_input = f"{text_title} {text_abstract}".strip()
    if not text_input:
        print(f"Skipping row {i} due to empty combined text_input for title: {text_title[:30]}")
        continue

    print(f"\nProcessing article {i+1}/{num_items_to_process} in data_pickler: {text_title[:60]}...")

    try:
        shap_explanations_list = explainer([text_input])
    except Exception as e:
        print(f"  Error during SHAP explanation for article {text_title[:60]}: {e}")
        continue

    prediction = "Error"
    try:
        prediction_result = sent_analyzer(text_input)
        if isinstance(prediction_result, list) and len(prediction_result) > 0:
            top_pred_dict = prediction_result[0]
            if isinstance(top_pred_dict, list) and len(top_pred_dict) > 0: top_pred_dict = top_pred_dict[0]
            if isinstance(top_pred_dict, dict):
                prediction_label = top_pred_dict.get("label")
                prediction = label_map.get(prediction_label, prediction_label)
    except Exception as e:
        print(f"  Error during sentiment analysis for prediction for article {text_title[:60]}: {e}")

    parsed_tokens: List[str] = []
    parsed_scores_per_class: List[List[float]] = []
    current_output_names: List[str] = ["Irrelevant", "Relevant"]

    if shap_explanations_list and isinstance(shap_explanations_list, list) and len(shap_explanations_list) > 0:
        explanation_obj = shap_explanations_list[0]
        data_attr = getattr(explanation_obj, 'data', None)
        values_attr = getattr(explanation_obj, 'values', None)
        output_names_attr = getattr(explanation_obj, 'output_names', None)

        print(f"  data_pickler: Raw SHAP data_attr type: {type(data_attr)}, shape/len: {getattr(data_attr, 'shape', len(data_attr) if hasattr(data_attr, '__len__') else 'N/A')}")
        print(f"  data_pickler: Raw SHAP values_attr type: {type(values_attr)}, shape: {getattr(values_attr, 'shape', 'N/A')}, dtype: {getattr(values_attr, 'dtype', 'N/A')}")
        
        if output_names_attr is not None:
            current_output_names = [str(name) for name in (output_names_attr.tolist() if isinstance(output_names_attr, np.ndarray) else output_names_attr)]
        num_expected_classes = len(current_output_names)

        # Attempt 1: Flatter structure (expected from Text masker)
        if (isinstance(data_attr, np.ndarray) and data_attr.ndim == 1 and
            isinstance(values_attr, np.ndarray) and values_attr.ndim == 2 and
            data_attr.shape[0] == values_attr.shape[0] and
            values_attr.shape[1] == num_expected_classes):
            print(f"    data_pickler: Parsing as flat SHAP structure (tokens: {data_attr.shape[0]}).")
            parsed_tokens = [str(token) for token in data_attr.tolist()]
            parsed_scores_per_class = [[float(score) for score in token_scores] for token_scores in values_attr.tolist()]
        
        # Attempt 2: Complex nested structure (fallback, if Text masker output is still complex)
        elif (isinstance(data_attr, tuple) and
              isinstance(values_attr, np.ndarray) and values_attr.dtype == 'object' and
              len(data_attr) == len(values_attr)):
            print(f"    data_pickler: Parsing as complex (nested) SHAP structure (parts: {len(data_attr)}).")
            for token_part_array, value_part_array in zip(data_attr, values_attr):
                if (isinstance(token_part_array, np.ndarray) and token_part_array.ndim == 1 and len(token_part_array) >= 2 and # Check for at least 2 elements for token
                    isinstance(value_part_array, np.ndarray) and value_part_array.ndim == 2 and
                    value_part_array.shape[0] >= 2 and value_part_array.shape[1] == num_expected_classes): # Check for at least 2 rows for scores
                    actual_token = str(token_part_array[1]) # Assuming token is at index 1
                    token_scores = [float(v) for v in value_part_array[1, :].tolist()] # Assuming scores are in row at index 1
                    parsed_tokens.append(actual_token)
                    parsed_scores_per_class.append(token_scores)
        else:
            print(f"    data_pickler: Warning - SHAP data not in a recognized flat or complex format.")

    if not parsed_tokens:
         print(f"    data_pickler: No tokens were parsed for {text_title[:60]}. Check SHAP object structure.")

    processed_shap_data = {
        "tokens": parsed_tokens,
        "scores_per_class": parsed_scores_per_class,
        "output_class_names": current_output_names
    }

    articles_to_process.append({
        "title": text_title,
        "abstract": text_abstract,
        "processed_shap_values": processed_shap_data, # Key for the simplified data
        "verdict": prediction
    })
    print(f"  data_pickler: Finished. Tokens extracted: {len(parsed_tokens)}")

output_filename = f"shap/shaps_{topic}.pkl"
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
with open(output_filename, "wb") as f:
    pickle.dump(articles_to_process, f)
print(f"\nSuccessfully processed {len(articles_to_process)} articles and saved to {output_filename}")
