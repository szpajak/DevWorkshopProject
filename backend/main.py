from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import glob
import os
from typing import List, Dict, Any, Optional
import numpy as np
import re

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
    explained_title_parts: List[TokenShapValue]
    explained_abstract_parts: List[TokenShapValue]
    top_words: List[TokenShapValue]

class TopicDataResponse(BaseModel):
    topic_name: str
    articles: List[ArticleShapData]

def get_relevant_class_index(output_names: List[Any], target_class: str = "Relevant") -> Optional[int]:
    """Finds the index of the target class in the list of output names."""
    processed_names = [str(name) for name in output_names]
    try:
        return processed_names.index(target_class)
    except ValueError:
        if target_class == "Relevant" and "LABEL_1" in processed_names:
            return processed_names.index("LABEL_1")
        if len(processed_names) == 2:
            return 1
    return None


def process_raw_shap_explanation(explanation_obj: Any) -> Dict[str, Any]:
    """
    Processes a SHAP explanation object, expecting word-level tokens.
    """
    if explanation_obj is None:
        return {"tokens": [], "scores_per_class": [], "output_class_names": []}

    tokens = getattr(explanation_obj, 'data', [])
    scores_per_class = getattr(explanation_obj, 'values', np.array([]))
    output_class_names = getattr(explanation_obj, 'output_names', [])

    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    if isinstance(output_class_names, np.ndarray):
        output_class_names = output_class_names.tolist()

    if not isinstance(tokens, list) or not isinstance(scores_per_class, np.ndarray) or scores_per_class.ndim != 2:
        return {"tokens": [], "scores_per_class": [], "output_class_names": []}
    
    if len(tokens) != scores_per_class.shape[0]:
        return {"tokens": [], "scores_per_class": [], "output_class_names": []}

    return {
        "tokens": [str(t) for t in tokens],
        "scores_per_class": scores_per_class.tolist(),
        "output_class_names": [str(n) for n in output_class_names]
    }


def create_explained_parts(processed_shap_data: Dict[str, Any], relevant_idx: int) -> List[TokenShapValue]:
    """Helper to create a list of TokenShapValue from processed SHAP data."""
    explained_parts: List[TokenShapValue] = []
    tokens: List[str] = processed_shap_data.get("tokens", [])
    scores_per_class: List[List[float]] = processed_shap_data.get("scores_per_class", [])

    if not tokens or not scores_per_class:
        return []

    for i, token_str in enumerate(tokens):
        try:
            if i < len(scores_per_class) and relevant_idx < len(scores_per_class[i]):
                relevant_score = float(scores_per_class[i][relevant_idx])
                explained_parts.append(TokenShapValue(token=token_str, value=relevant_score))
            else:
                explained_parts.append(TokenShapValue(token=token_str, value=0.0))
        except (TypeError, ValueError) as e:
            print(f"    Error processing score for token '{token_str}': {e}. Using 0.0.")
            explained_parts.append(TokenShapValue(token=token_str, value=0.0))
    return explained_parts


def process_article_for_api(article_data_raw: Dict[str, Any]) -> ArticleShapData:
    article_title = str(article_data_raw.get("title", "N/A"))
    article_abstract = str(article_data_raw.get("abstract", ""))
    article_verdict = str(article_data_raw.get("verdict", "N/A"))

    print(f"\nProcessing article for API: {article_title[:60]}")

    raw_shap_title = article_data_raw.get("shap_title_values")
    raw_shap_abstract = article_data_raw.get("shap_abstract_values")

    processed_title_data = process_raw_shap_explanation(raw_shap_title)
    processed_abstract_data = process_raw_shap_explanation(raw_shap_abstract)

    # Use output names from title or abstract if available, they should be the same
    output_class_names = processed_title_data.get("output_class_names") or processed_abstract_data.get("output_class_names", [])
    
    relevant_idx = get_relevant_class_index(output_class_names, "Relevant")
    if relevant_idx is None:
        print(f"  Error: Could not determine 'Relevant' class index from {output_class_names}. Cannot process SHAP values.")
        return ArticleShapData(title=article_title, abstract=article_abstract, verdict=article_verdict, explained_title_parts=[], explained_abstract_parts=[], top_words=[])

    explained_title_parts = create_explained_parts(processed_title_data, relevant_idx)
    explained_abstract_parts = create_explained_parts(processed_abstract_data, relevant_idx)

    # Combine all parts for top words calculation
    all_explained_parts = explained_title_parts + explained_abstract_parts
    
    top_words: List[TokenShapValue] = []
    if all_explained_parts:
        # Filter out non-alphabetic tokens (like punctuation, numbers) and empty/whitespace tokens.
        # A token is considered a word if it contains at least one letter.
        word_tokens_for_sorting = [
            part for part in all_explained_parts if part.token.strip() and re.search(r'[a-zA-Z]', part.token)
        ]
        
        # Sort based on the article's verdict
        if article_verdict == "Relevant":
            # For "Relevant" articles, find words with the highest positive scores
            word_importances = sorted(word_tokens_for_sorting, key=lambda x: x.value, reverse=True)
        elif article_verdict == "Irrelevant":
            # For "Irrelevant" articles, find words with the most negative scores
            word_importances = sorted(word_tokens_for_sorting, key=lambda x: x.value, reverse=False)
        else:
            # Fallback for any other verdict: sort by absolute impact
            word_importances = sorted(word_tokens_for_sorting, key=lambda x: abs(x.value), reverse=True)

        # Ensure uniqueness (case-insensitive) and get top 5
        unique_top_words: List[TokenShapValue] = []
        seen_tokens = set()
        for part in word_importances:
            if len(unique_top_words) >= 5:
                break
            # Normalize token for uniqueness check (lowercase)
            normalized_token = part.token.lower()
            if normalized_token not in seen_tokens:
                seen_tokens.add(normalized_token)
                unique_top_words.append(part)
        
        top_words = unique_top_words

    print(f"  Successfully processed for API. Title parts: {len(explained_title_parts)}, Abstract parts: {len(explained_abstract_parts)}, Top words: {len(top_words)}")
    return ArticleShapData(
        title=article_title,
        abstract=article_abstract,
        verdict=article_verdict,
        explained_title_parts=explained_title_parts,
        explained_abstract_parts=explained_abstract_parts,
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
    # Find all matching SHAP files for the topic
    file_pattern = os.path.join(SHAP_DIR, f"shaps_{topic_name}*.pkl")
    matching_files = sorted(glob.glob(file_pattern))

    if not matching_files:
        raise HTTPException(status_code=404, detail=f"No SHAP data files found for topic '{topic_name}' in path {SHAP_DIR}.")

    shaps_list_for_topic: List[Dict[str, Any]] = []

    for file_path in matching_files:
        abs_file_path = os.path.abspath(file_path)
        print(f"Loading SHAP data from: {abs_file_path}")

        try:
            with open(file_path, "rb") as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, list):
                    shaps_list_for_topic.extend(loaded_data)
                else:
                    print(f"  Warning: Data in {file_path} is not a list. Skipping.")
        except Exception as e:
            print(f"  Error loading file {file_path}: {e}. Skipping.")
            continue

    if not shaps_list_for_topic:
        raise HTTPException(status_code=500, detail=f"No valid SHAP data found for topic '{topic_name}'.")

    processed_articles_for_api: List[ArticleShapData] = []
    for i, article_data_raw in enumerate(shaps_list_for_topic):
        if not isinstance(article_data_raw, dict):
            print(f"  Warning: Article data at index {i} in topic '{topic_name}' is not a dict. Skipping.")
            continue
        processed_articles_for_api.append(process_article_for_api(article_data_raw))

    return TopicDataResponse(topic_name=topic_name, articles=processed_articles_for_api)
