import os
import warnings
import sys

import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn import preprocessing
import logging
from urllib.parse import parse_qsl, urljoin, urlparse
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

data_path = "train_LZdllcl.csv"

def eval_metrics(actual, pred):
    F1_score = f1_score(actual, pred)
    rocauc=  roc_auc_score(actual, pred)
    return F1_score,rocauc


def load_data(data_path):
    data = pd.read_csv(data_path)
    data.sample(10)
    data.isnull().sum()
    mode_edu = data['education'].mode()[0]
    mode_prev_year = data['previous_year_rating'].mode()[0]
    data['education'] = data['education'].fillna(mode_edu)
    data['previous_year_rating'] = data['previous_year_rating'].fillna(mode_prev_year)
    le = preprocessing.LabelEncoder()
    data['education'] = le.fit_transform(data['education'])
    data['gender'] = le.fit_transform(data['gender'])
    data['recruitment_channel'] = le.fit_transform(data['recruitment_channel'])
    data['region'] = le.fit_transform(data['region'])
    data['department'] = le.fit_transform(data['department'])
    train, test = train_test_split(data)
    train_x = train.drop(["is_promoted"], axis=1)
    test_x = test.drop(["is_promoted"], axis=1)
    train_y = train[["is_promoted"]]
    test_y = test[["is_promoted"]]
    return train_x, train_y, test_x, test_y


def train(n_estimators=300, max_depth=5):
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    data_path = "train_LZdllcl.csv"
    train_x, train_y, test_x, test_y = load_data(data_path)
    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(train_x, train_y)

        predicted_qualities = rf.predict(test_x)
        (F1_score, rocauc) = eval_metrics(test_y, predicted_qualities)

        print("Randomforest model (n_estimators=%f, max_depth=%f):" % (n_estimators, max_depth))
        print("  F1 score: %s" % F1_score)
        print("  Roc_auc: %s" % rocauc)

        mlflow.log_param(key="n_estimators", value=n_estimators)
        mlflow.log_param(key="max_depth", value=max_depth)
        mlflow.log_metric("F1_score", F1_score)
        mlflow.log_metric("Roc_auc", rocauc)
        mlflow.log_artifact(data_path)
        print("Save to: {}".format(mlflow.get_artifact_uri()))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
           mlflow.sklearn.log_model(rf, "model", registered_model_name="HR_MODEL")
        else:
           mlflow.sklearn.log_model(rf, "model")



train(1250,7)



