from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import glob
import os
from typing import List, Dict, Any, Optional
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SHAP_DIR = "../shap/" 

class TokenShapValue(BaseModel):
    token: str
    value: float

class ArticleShapData(BaseModel):
    title: str
    abstract: str
    verdict: str
    explained_text_parts: List[TokenShapValue]
    top_words: List[TokenShapValue]

class TopicDataResponse(BaseModel):
    topic_name: str
    articles: List[ArticleShapData]

def get_relevant_class_index(output_names: List[Any], target_class: str = "Relevant") -> Optional[int]:
    print(f"    get_relevant_class_index: Received output_names type {type(output_names)}, content: {str(output_names)[:100]}")
    processed_output_names_for_index: List[str] = []
    if isinstance(output_names, np.ndarray):
        processed_output_names_for_index = [str(name) for name in output_names.tolist()]
    elif isinstance(output_names, list):
        processed_output_names_for_index = [str(name) for name in output_names]
    else:
        print(f"    get_relevant_class_index: output_names is not list or ndarray. Defaulting to index 1 if binary.")
        # Cannot determine length reliably if not list/ndarray
        return 1 if target_class == "Relevant" else 0 # Simplified fallback for now

    print(f"    get_relevant_class_index: Processed names for index search: {processed_output_names_for_index}")
    try:
        return processed_output_names_for_index.index(target_class)
    except ValueError:
        print(f"    Warning: '{target_class}' not found in processed_output_names: {processed_output_names_for_index}.")
        if target_class == "Relevant" and "LABEL_1" in processed_output_names_for_index:
            idx = processed_output_names_for_index.index("LABEL_1")
            print(f"      Found 'LABEL_1' at index {idx} as fallback for 'Relevant'.")
            return idx
        if len(processed_output_names_for_index) == 2: 
            print(f"      Assuming index 1 for '{target_class}' (binary fallback).")
            return 1
        if len(processed_output_names_for_index) == 1: 
            print(f"      Assuming index 0 for '{target_class}' (unary fallback).")
            return 0
    print(f"    Error: Could not definitively determine index for '{target_class}' from: {processed_output_names_for_index}")
    return None


def process_raw_shap_explanation(explanation_obj: Any) -> Dict[str, Any]:
    print("  process_raw_shap_explanation: Entered function.")
    parsed_tokens: List[str] = []
    parsed_scores_per_class: List[List[float]] = []
    current_output_names: List[str] = ["Irrelevant", "Relevant"] # Default

    if explanation_obj is None:
        print("  process_raw_shap_explanation: Received None explanation_obj. Returning empty.")
        return {"tokens": [], "scores_per_class": [], "output_class_names": current_output_names}

    print(f"  process_raw_shap_explanation: explanation_obj type: {type(explanation_obj)}")

    data_attr = getattr(explanation_obj, 'data', None)
    values_attr = getattr(explanation_obj, 'values', None)
    output_names_attr = getattr(explanation_obj, 'output_names', None)

    print(f"    Initial data_attr type: {type(data_attr)}")
    if data_attr is not None: print(f"    Initial data_attr content (first 5 if tuple/list else N/A): {str(data_attr[:5]) if isinstance(data_attr, (tuple,list)) and len(data_attr)>0 else str(data_attr)[:200]}")
    
    print(f"    Initial values_attr type: {type(values_attr)}")
    if values_attr is not None: print(f"    Initial values_attr shape (if ndarray): {values_attr.shape if isinstance(values_attr, np.ndarray) else 'N/A'}, dtype: {values_attr.dtype if hasattr(values_attr, 'dtype') else 'N/A'}")
    
    print(f"    Initial output_names_attr type: {type(output_names_attr)}, content: {str(output_names_attr)[:200]}")

    if output_names_attr is not None:
        if isinstance(output_names_attr, np.ndarray):
            current_output_names = [str(name) for name in output_names_attr.tolist()]
        elif isinstance(output_names_attr, list):
            current_output_names = [str(name) for name in output_names_attr]
        else:
            print(f"    Warning: output_names_attr is of unexpected type: {type(output_names_attr)}. Using default: {current_output_names}")
    else:
        print(f"    Warning: output_names_attr is None. Using default: {current_output_names}")
            
    num_expected_classes = len(current_output_names)
    print(f"    Determined num_expected_classes: {num_expected_classes} from current_output_names: {current_output_names}")

    # Condition 1: Complex SHAP structure (tuple of arrays for data, object array of arrays for values)
    cond1_data_is_tuple = isinstance(data_attr, tuple)
    cond1_values_is_ndarray = isinstance(values_attr, np.ndarray)
    cond1_values_dtype_is_object = values_attr.dtype == 'object' if cond1_values_is_ndarray else False
    cond1_lengths_match = len(data_attr) == len(values_attr) if cond1_data_is_tuple and cond1_values_is_ndarray else False
    
    print(f"    Complex structure check: data_is_tuple={cond1_data_is_tuple}, values_is_ndarray={cond1_values_is_ndarray}, values_dtype_is_object={cond1_values_dtype_is_object}, lengths_match={cond1_lengths_match}")

    if cond1_data_is_tuple and cond1_values_is_ndarray and cond1_values_dtype_is_object and cond1_lengths_match:
        print(f"    Attempting to parse complex SHAP structure: {len(data_attr)} token parts.")
        processed_count = 0
        for idx, (token_part_array, value_part_array) in enumerate(zip(data_attr, values_attr)):
            # Inner conditions for each part
            c_token_np = isinstance(token_part_array, np.ndarray)
            c_token_ndim1 = token_part_array.ndim == 1 if c_token_np else False
            c_token_len3 = len(token_part_array) == 3 if c_token_np and c_token_ndim1 else False
            
            c_value_np = isinstance(value_part_array, np.ndarray)
            c_value_ndim2 = value_part_array.ndim == 2 if c_value_np else False
            c_value_shape0_3 = value_part_array.shape[0] == 3 if c_value_np and c_value_ndim2 else False
            c_value_shape1_ok = value_part_array.shape[1] == num_expected_classes if c_value_np and c_value_ndim2 else False
            
            if (c_token_np and c_token_ndim1 and c_token_len3 and
                c_value_np and c_value_ndim2 and c_value_shape0_3 and c_value_shape1_ok):
                actual_token = str(token_part_array[1])
                token_scores = [float(v) for v in value_part_array[1, :].tolist()]
                parsed_tokens.append(actual_token)
                parsed_scores_per_class.append(token_scores)
                processed_count +=1
            else:
                print(f"      Skipping part {idx}: Condition failed.")
                print(f"        token_part: type={type(token_part_array)}, ndim={token_part_array.ndim if c_token_np else 'N/A'}, len={len(token_part_array) if c_token_np and c_token_ndim1 else 'N/A'}")
                print(f"        value_part: type={type(value_part_array)}, ndim={value_part_array.ndim if c_value_np else 'N/A'}, shape={value_part_array.shape if c_value_np and c_value_ndim2 else 'N/A'}")
                print(f"        Conditions: c_token_np={c_token_np}, c_token_ndim1={c_token_ndim1}, c_token_len3={c_token_len3}")
                print(f"                      c_value_np={c_value_np}, c_value_ndim2={c_value_ndim2}, c_value_shape0_3={c_value_shape0_3}, c_value_shape1_ok={c_value_shape1_ok} (expected classes: {num_expected_classes})")
        print(f"    Finished complex structure parsing. Processed {processed_count} parts.")

    # Condition 2: Flatter SHAP structure (list/array for data, 2D array for values)
    elif isinstance(data_attr, (list, np.ndarray)) and isinstance(values_attr, np.ndarray) and values_attr.ndim == 2:
        print(f"    Attempting to parse flat SHAP structure. Data len: {len(data_attr)}, Values shape: {values_attr.shape}")
        if len(data_attr) == values_attr.shape[0] and values_attr.shape[1] == num_expected_classes:
            parsed_tokens = [str(t) for t in data_attr]
            parsed_scores_per_class = [[float(v) for v in row] for row in values_attr.tolist()]
            print(f"      Successfully parsed flat structure. Tokens: {len(parsed_tokens)}")
        else:
            print(f"      Warning: Flat SHAP data length or class count mismatch. Tokens: {len(data_attr)}, Values shape: {values_attr.shape}, Expected classes: {num_expected_classes}")
    else:
        print(f"    Warning: Raw SHAP .data or .values not in a recognized format for either complex or flat parsing.")

    print(f"  process_raw_shap_explanation: Returning {len(parsed_tokens)} tokens and {len(parsed_scores_per_class)} score sets.")
    return {
        "tokens": parsed_tokens,
        "scores_per_class": parsed_scores_per_class,
        "output_class_names": current_output_names
    }

# process_article_for_api and other endpoints remain the same as in the previous good backend version
def process_article_for_api(article_data_raw: Dict[str, Any]) -> ArticleShapData:
    article_title = str(article_data_raw.get("title", "N/A"))
    article_abstract = str(article_data_raw.get("abstract", ""))
    article_verdict = str(article_data_raw.get("verdict", "N/A"))

    print(f"\nProcessing article for API: {article_title[:60]}")

    raw_shap_explanation = article_data_raw.get("shap_values") 
    processed_shap_data = process_raw_shap_explanation(raw_shap_explanation)

    tokens: List[str] = processed_shap_data.get("tokens", [])
    scores_per_class: List[List[float]] = processed_shap_data.get("scores_per_class", [])
    output_class_names: List[str] = processed_shap_data.get("output_class_names", ["Irrelevant", "Relevant"])

    if not tokens or not scores_per_class: # This is the warning user saw
        print(f"  Warning: Parsed tokens or scores_per_class are empty for article: {article_title}. Tokens: {len(tokens)}, Scores: {len(scores_per_class)}")
        return ArticleShapData(title=article_title, abstract=article_abstract, verdict=article_verdict, explained_text_parts=[], top_words=[])

    if len(tokens) != len(scores_per_class):
        print(f"  Error: Mismatch between parsed token count ({len(tokens)}) and scores count ({len(scores_per_class)})")
        explained_text_parts = [TokenShapValue(token=t, value=0.0) for t in tokens] if tokens else []
        return ArticleShapData(title=article_title, abstract=article_abstract, verdict=article_verdict, explained_text_parts=explained_text_parts, top_words=[])

    relevant_idx = get_relevant_class_index(output_class_names, "Relevant")

    if relevant_idx is None:
        print(f"  Error: Could not determine 'Relevant' class index from {output_class_names}")
        explained_text_parts = [TokenShapValue(token=t, value=0.0) for t in tokens]
        return ArticleShapData(title=article_title, abstract=article_abstract, verdict=article_verdict, explained_text_parts=explained_text_parts, top_words=[])

    explained_text_parts: List[TokenShapValue] = []
    for i, token_str in enumerate(tokens):
        try:
            if relevant_idx < len(scores_per_class[i]):
                relevant_score = float(scores_per_class[i][relevant_idx])
                explained_text_parts.append(TokenShapValue(token=token_str, value=relevant_score))
            else:
                print(f"    Warning: relevant_idx {relevant_idx} out of bounds for scores of token '{token_str}' (scores: {scores_per_class[i]}). Using 0.0.")
                explained_text_parts.append(TokenShapValue(token=token_str, value=0.0))
        except (IndexError, TypeError, ValueError) as e:
            print(f"    Error processing score for token '{token_str}': {e}. Scores: {scores_per_class[i] if i < len(scores_per_class) else 'N/A'}. Using 0.0.")
            explained_text_parts.append(TokenShapValue(token=token_str, value=0.0))
            
    top_words: List[TokenShapValue] = []
    if explained_text_parts:
        word_importances = sorted(explained_text_parts, key=lambda x: abs(x.value), reverse=True)
        top_words = word_importances[:5]
    
    print(f"  Successfully processed for API. Explained parts: {len(explained_text_parts)}, Top words: {len(top_words)}")
    return ArticleShapData(
        title=article_title,
        abstract=article_abstract,
        verdict=article_verdict,
        explained_text_parts=explained_text_parts,
        top_words=top_words,
    )

@app.get("/topics", response_model=List[str])
async def list_topics():
    try:
        topic_files = glob.glob(os.path.join(SHAP_DIR, "shaps_*.pkl"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing SHAP directory: {str(e)}")
    if not topic_files:
        abs_shap_dir = os.path.abspath(SHAP_DIR)
        raise HTTPException(status_code=404, detail=f"No SHAP topic files found in {abs_shap_dir}. Searched for 'shaps_*.pkl'.")
    topics = [os.path.basename(f_path)[len("shaps_"):-len(".pkl")] for f_path in topic_files if os.path.basename(f_path).startswith("shaps_") and os.path.basename(f_path).endswith(".pkl")]
    if not topics:
        raise HTTPException(status_code=404, detail="No topics extracted from found .pkl files.")
    return topics

@app.get("/shap_data/{topic_name}", response_model=TopicDataResponse)
async def get_shap_data_for_topic(topic_name: str):
    file_path = os.path.join(SHAP_DIR, f"shaps_{topic_name}.pkl")
    abs_file_path = os.path.abspath(file_path)
    print(f"Attempting to load SHAP data from: {abs_file_path}")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"SHAP data file for topic '{topic_name}' not found at {abs_file_path}.")

    try:
        with open(file_path, "rb") as f:
            shaps_list_for_topic = pickle.load(f)
    except Exception as e:
        print(f"Error loading or unpickling SHAP data for {abs_file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading or unpickling SHAP data for '{topic_name}'. Error: {e}")

    if not isinstance(shaps_list_for_topic, list):
        raise HTTPException(status_code=500, detail=f"SHAP data file for '{topic_name}' does not contain a list. Type: {type(shaps_list_for_topic)}")

    processed_articles_for_api: List[ArticleShapData] = []
    for i, article_data_raw in enumerate(shaps_list_for_topic):
        if not isinstance(article_data_raw, dict):
            print(f"  Warning: Article data at index {i} for topic '{topic_name}' is not a dict. Skipping.")
            continue
        processed_articles_for_api.append(process_article_for_api(article_data_raw))
            
    return TopicDataResponse(topic_name=topic_name, articles=processed_articles_for_api)