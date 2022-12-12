#!/usr/bin/env python3

import os
import glob
from pickle import load

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_training(path):
    """
    Compile all training datasets in path and concatenate into one dataframe.
    path (str): The directory containing training sets.
    """
    datasets = glob.iglob(os.path.join(path, "*.csv"))
    dataset_list = []
    for dataset in datasets:
        data = pd.read_csv(dataset, index_col="UUID")
        dataset_list.append(data)
    print(f"Training on {len(dataset_list)} datasets.")
    df = pd.concat(dataset_list, axis=0)
    return df

pipe = Pipeline((("scaler", StandardScaler()), ("pca", PCA())))

data = [
    build_training(r"../Data/Training"),
    pd.read_csv("../Data/Unidentified/08-17-2022_Data.csv"),
]

response = []
for i, dataset in enumerate(data):
    dataset.columns = (
        dataset.columns.str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    data[i] = dataset.select_dtypes(float)
    data[i].drop(["ch2-ch1_ratio", "aspect_ratio"], axis=1, inplace=True)

breakpoint()
training_scaled = pipe.fit_transform(data[0])
test_scaled = pipe.transform(data[1])

path = r"../Data/Models"
files = [os.path.join(path, file) for file in os.listdir(path)]
with open(max(files, key=os.path.getctime), "rb") as file:
    models = load(file)

model = models.best_estimator_
pred = model.predict(test_scaled)
pred_prob = model.predict_proba(test_scaled)
