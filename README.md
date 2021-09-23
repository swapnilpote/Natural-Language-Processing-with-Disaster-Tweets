# Natural Language Processing with Disaster Tweets
This is Kaggle competition on Text Classification based Disaster and Non Disaster categories.

Competition link => https://www.kaggle.com/c/nlp-getting-started/

### Steps to follow
1. Initiate Git and DVC for code and data versioning.
2. Download data from Kaggle using official API.
3. Sync data with DVC to maintain versioning.


### DVC pipeline
1. dvc run -n prepare -p prepare.seed,prepare.split -d src\prepare.py -d data\train.csv -d data\test.csv -o data\prepared python src\prepare.py data\train.csv data\test.csv
2. dvc run --force -n featurize -p featurize.max_features,featurize.ngrams -d src\featurization.py -d data\prepared -o data\features python src\featurization.py data\prepared data\features