import os
import sys
import yaml

# Data Science imports
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(open("params.yaml"))["train"]

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train.py prepared features model\n")
    sys.exit(1)


seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]

in_prep_file = sys.argv[1]
in_feat_file = sys.argv[2]
op_file = sys.argv[3]


def model_train(in_prep_path: str, in_feat_path: str, op_file_path: str) -> None:
    with open(os.path.join(in_feat_path, "train.pkl"), "rb") as fd:
        X = joblib.load(fd)

    df = pd.read_csv(os.path.join(in_prep_path, "train.csv"))
    y = df["target"].values

    sys.stderr.write("X matrix size {}\n".format(X.shape))
    sys.stderr.write("Y matrix size {}\n".format(y.shape))

    clf = RandomForestClassifier(
        n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
    )

    clf.fit(X, y)

    with open(op_file_path, "wb") as fd:
        joblib.dump(clf, fd)


if __name__ == "__main__":
    model_train(in_prep_file, in_feat_file, op_file)
