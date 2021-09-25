import os
import sys
import json
import math

# Data Science imports
import joblib
import pandas as pd
import sklearn.metrics as metrics

if len(sys.argv) != 7:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py model prepared features scores prc roc\n")
    sys.exit(1)

model_file = sys.argv[1]
in_prep_file = os.path.join(sys.argv[2], "valid.csv")
in_feat_file = os.path.join(sys.argv[3], "valid.pkl")
scores_file = sys.argv[4]
prc_file = sys.argv[5]
roc_file = sys.argv[6]


def model_score(in_prep_path: str, in_feat_path: str, model_file_path: str) -> None:
    """Perform model evaluation on validation/hold out data to check accuracy.

    Args:
        in_prep_path (str): Prepared data file to extract labels.
        in_feat_path (str): Featurized data file to extract numpy array.
        model_file_path (str): Model file path.
    """
    with open(model_file_path, "rb") as f:
        model = joblib.load(f)

    with open(in_feat_path, "rb") as f:
        X = joblib.load(f)

    df = pd.read_csv(in_prep_path)
    y = df["target"].values

    predictions = model.predict(X)
    precision, recall, prc_thresholds = metrics.precision_recall_curve(y, predictions)
    fpr, tpr, roc_thresholds = metrics.roc_curve(y, predictions)

    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)

    with open(scores_file, "w") as fd:
        json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]

    with open(prc_file, "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": float(p), "recall": float(r), "threshold": float(t)}
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
        )

    with open(roc_file, "w") as fd:
        json.dump(
            {
                "roc": [
                    {"fpr": float(fp), "tpr": float(tp), "threshold": float(t)}
                    for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                ]
            },
            fd,
            indent=4,
        )

    return None


if __name__ == "__main__":
    model_score(in_prep_file, in_feat_file, model_file)
