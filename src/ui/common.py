import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils import resample
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
        max_length=1000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.X = X

        topics = X["title"].astype(str).tolist()
        abstracts = X["abstract"].astype(str).tolist()

        encodings = self.tokenizer(
            topics,
            abstracts,
            truncation=True,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.token_type_ids = encodings["token_type_ids"]
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.y[idx],
        }


class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(2, 1)
        self.config = AutoConfig.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        x = outputs.logits  # Access the logits attribute for classification
        x = self.dropout(x)
        x = self.fc(x)

        return x.squeeze(-1)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_Xy(topic, df):
    df_topic = df[df["topic_id"] == topic]
    df_topic.dropna(inplace=True)
    df_topic.reset_index(drop=True, inplace=True)

    X = df_topic.drop(columns=["relevance", "topic_id", "PID"], axis=1)
    y = df_topic["relevance"]

    return X, y


def preprocess(topic, df):
    set_seed(42)

    df_majority = df[df["relevance"] == 0]
    df_minority = df[df["relevance"] == 1]

    X_maj, y_maj = split_Xy(topic, df_majority)
    X_min, y_min = split_Xy(topic, df_minority)

    X_rsmpld, y_resmpld = resample(
        X_maj, y_maj, replace=False, n_samples=len(y_min), random_state=42
    )

    X_balanced = pd.concat([X_rsmpld, X_min])
    y_balanced = pd.concat([y_resmpld, y_min])

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, stratify=y_balanced, test_size=0.25, random_state=42
    )

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def load_model_from_file(topic):
    model = BERTClassifier()
    model.load_state_dict(torch.load(f"models/bert_classifier_{topic}_balanced.pt", map_location=torch.device("cpu")))
    return model


def get_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


def get_top_topics(df):
    return (
        df.groupby("topic_id")
        .sum("relevance")[df.groupby("topic_id").sum("relevance")["relevance"] >= 70]
        .sort_values(by="relevance", ascending=False)
        .index
    )


def get_raw_data():
    DATA_PATH = "data/preprocessed"
    TRAIN_PATH = DATA_PATH + "/train.csv"
    TEST_PATH = DATA_PATH + "/test.csv"
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    return (df_train, df_test)
