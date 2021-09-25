import os
import re
import sys
import yaml

# Data Science imports
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer


params = yaml.safe_load(open("params.yaml"))["prepare"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py traindata-file testdata-file\n")
    sys.exit(1)

# Validation data set split ratio
split = params["split"]
seed = params["seed"]

train_input = sys.argv[1]
test_input = sys.argv[2]
output_train = os.path.join("data", "prepared", "train.csv")
output_valid = os.path.join("data", "prepared", "valid.csv")
output_test = os.path.join("data", "prepared", "test.csv")


def clean_text(text: str) -> str:
    """[summary]

    Args:
        text (str): [description]

    Returns:
        str: [description]
    """
    symbols = re.findall("[^a-zA-Z0-9#@'\\s]", text)

    for sym in symbols:
        text = text.replace(sym, f" {sym} ")

    return text.lower()


def main(file_path: str, data_type: str = "train") -> None:
    """[summary]

    Args:
        file_path (str): [description]
        data_type (str, optional): [description]. Defaults to "train".
    """
    df = pd.read_csv(file_path, encoding="utf8")
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    try:
        # Drop unnecessary columns
        df = df.drop(["keyword", "location"], axis=1)

        # Clean text by separating special characteres from words
        df["text"] = df["text"].apply(clean_text)
        df["text"] = df["text"].apply(lambda x: " ".join(x.split()))

        # Generating root word using lemmatization technique
        lemmatizer = WordNetLemmatizer()
        df["text"] = df["text"].apply(
            lambda x: " ".join([lemmatizer.lemmatize(w) for w in x.split()])
        )

        if data_type == "train":
            train, valid = train_test_split(df, test_size=split, random_state=seed)
            train.to_csv(output_train, index=False)
            valid.to_csv(output_valid, index=False)
        else:
            df.to_csv(output_test, index=False)

    except Exception as ex:
        sys.stderr.write(ex)


if __name__ == "__main__":
    main(train_input)
    main(test_input, data_type="test")
