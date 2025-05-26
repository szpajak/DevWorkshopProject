import shap
import transformers
import pickle

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

shaps = {}


sent_analyzer = transformers.pipeline(
    "sentiment-analysis",
    tokenizer=tokenizer,
    top_k=10,
    model=model.bert,
)
explainer = shap.Explainer(sent_analyzer, output_names=["Irrelavant", "Relevant"])

for i in range(test_dataset.X.shape[0]):
    x = test_dataset.X.iloc[i]
    shap_values = explainer(x)
    shaps[x.title] = shap_values

    print(f"Pickled {x.title[0:20]}...")

shaps_file = open("shaps.pkl", "ab")
zakiszone = pickle.dumps(shaps, shaps_file)

shaps_file.close()
