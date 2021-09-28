# Natural Language Processing with Disaster Tweets
This is Kaggle competition on Text Classification based Disaster and Non Disaster categories.

Competition link => https://www.kaggle.com/c/nlp-getting-started/

### Steps to follow (https://dvc.org/doc/start/data-and-model-versioning)
1. Initiate Git and DVC for code and data versioning.
    - git init
    - dvc init (Store your data into appropriate folder)
2. Download data from Kaggle using official API.
3. Sync data with DVC to maintain versioning.
    - dvc add {folder/file path}
    - Make sure to check git command displayed on terminal after above command 
    - dvc remote add -d storage {url}
    - dvc push
4. If we want to sync prepared, featurize or model.pkl files then we have to repeat step 3.


### DVC pipeline
1. dvc run -n prepare -p prepare.seed,prepare.split -d src\prepare.py -d data\train.csv -d data\test.csv -o data\prepared python src\prepare.py data\train.csv data\test.csv
2. dvc run -n featurize -p featurize.max_features,featurize.ngrams -d src\featurization.py -d data\prepared -o data\features python src\featurization.py data\prepared data\features
3. dvc run -n train -p train.min_split,train.n_est,train.seed -d src\train.py -d data\prepared -d data\features -o model.pkl python src\train.py data\prepared data\features model.pkl
4. dvc run --force -n evaluate -d src\evaluate.py -d model.pkl -d data\prepared -d data\features -M metrics\scores.json --plots-no-cache metrics\prc.json --plots-no-cache metrics\roc.json python src\evaluate.py model.pkl data\prepared data\features metrics\scores.json metrics\prc.json metrics\roc.json
5. dvc metrics show
6. (Optional) dvc plots modify prc.json -x recall -y precision
7. (Optional) dvc plots modify roc.json -x fpr -y tpr
8. (Optional in case if you run either of step 6 or 7 or both) dvc plots show
    - Make sure to change "plot_metrics\prc_json" to "plot_metrics_prc_json" in div id section to see graph
    - Make sure to change "plot_metrics\roc_json" to "plot_metrics_roc_json" in div id section to see graph

Note: Use --force flag in case of re running any stages if it failed at initially.