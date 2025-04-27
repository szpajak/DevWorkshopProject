import os, io
import pandas as pd


def read_file_content(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content


def walk_over_files_with_ext(directory, extensions):
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                yield (file_path, read_file_content(file_path))


def walk_over_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            yield (file_path, read_file_content(file_path))


def parse_topic_file(content):
    extracted = {}
    lines = content.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Topic:"):
            extracted["Topic"] = line[len("Topic:") :].strip()
        elif line.startswith("Title:"):
            extracted["Title"] = line[len("Title:") :].strip()
        elif line.startswith("Objective:"):
            extracted["Objective"] = line[len("Objective:") :].strip()
        elif line.startswith("Objectives:"):
            extracted["Objective"] = line[len("Objectives:") :].strip()

    if not ("Topic" in extracted and "Title" in extracted and "Objective" in extracted):
        print(Warning(f"Could not parse topic {extracted['Topic']}"))
        return None

    return pd.DataFrame(
        [
            {
                "topic_id": extracted["Topic"],
                "topic_title": extracted["Title"],
                "topic_objective": extracted["Objective"],
            }
        ]
    )


def parse_qrels_file(content):
    try:
        data_io = io.StringIO(content)
        pids_df = pd.read_csv(
            data_io,
            sep="\s+",
            header=None,
            names=["topic_id", "irrelevant_1", "PID", "irrelevant_2"],
        )

        del pids_df["irrelevant_1"]
        del pids_df["irrelevant_2"]

        return pids_df
    except Exception as e:
        print("Could not pares qrel file", e)
        return None


def get_topics_pids_df(data_paths):
    topics_df = pd.DataFrame([], columns=["topic_id", "topic_title", "topic_objective"])
    qrels_df = pd.DataFrame([], columns=["topic_id", "PID"])

    for path in data_paths:
        for path, content in walk_over_files(path):
            if path.split("/")[-2] == "topics":
                topic_row = parse_topic_file(content)

                if topic_row is not None:
                    topics_df = pd.concat([topics_df, topic_row], ignore_index=True)

            if path.split("/")[-2] == "qrels":
                qrels_row = parse_qrels_file(content)

                if qrels_row is not None:
                    qrels_df = pd.concat([qrels_df, qrels_row], ignore_index=True)

    return pd.merge(topics_df, qrels_df, on="topic_id", how="left")
