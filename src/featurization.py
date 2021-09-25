import os
import sys
import yaml

# Data Science imports
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


params = yaml.safe_load(open("params.yaml"))["featurize"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
    sys.exit(1)

max_features = params["max_features"]
ngrams = params["ngrams"]

train_input = os.path.join(sys.argv[1], "train.csv")
valid_input = os.path.join(sys.argv[1], "valid.csv")
test_input = os.path.join(sys.argv[1], "test.csv")
bag_of_words_output = os.path.join(sys.argv[2], "bag_of_words.pkl")
train_output = os.path.join(sys.argv[2], "train.pkl")
valid_output = os.path.join(sys.argv[2], "valid.pkl")
test_output = os.path.join(sys.argv[2], "test.pkl")


def word_featurize(in_file_path: str, op_file_path: str) -> None:
    """This function generate features using TF-IDF method of scikit-learn

    Args:
        in_file_path (str): Input file path that should be featurized
        op_file_path (str): Output file path to store featurized data
    """
    df = pd.read_csv(in_file_path)
    os.makedirs(os.path.join("data", "features"), exist_ok=True)

    if in_file_path.split("\\")[-1].split(".")[0] == "train":
        bag_of_words = TfidfVectorizer(
            ngram_range=(1, ngrams), stop_words="english", max_features=max_features
        )

        tfidf = bag_of_words.fit_transform(df["text"].values)

        with open(bag_of_words_output, "wb") as f:
            joblib.dump(bag_of_words, f)
    else:
        with open(bag_of_words_output, "rb") as f:
            bag_of_words = joblib.load(f)
        tfidf = bag_of_words.transform(df["text"].values)

    with open(op_file_path, "wb") as f:
        joblib.dump(tfidf, f)

    return None


if __name__ == "__main__":
    for in_file, op_file in [
        [train_input, train_output],
        [valid_input, valid_output],
        [test_input, test_output],
    ]:
        word_featurize(in_file, op_file)
