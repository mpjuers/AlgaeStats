#!/usr/bin/env python3

import os
import glob
from pickle import load

from IPython import embed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pyhdfe as ph
from sklearn.linear_model import LogisticRegression

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

<<<<<<< HEAD
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
||||||| parent of 50d5f19 (Plots and scoring working.)
pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA())])


def create_data_frame(arr, transformer, index):
    return pd.DataFrame(
        arr,
        columns=transformer.get_feature_names_out() if transformer else None,
        index=index,
    )


def read_csv(path, *args, **kwargs):
    data = pd.read_csv(path)
    data.columns = (
        data.columns.str.replace(r" ", r"_")
        .str.replace(r"(", r"")
        .str.replace(r")", r"")
        .str.replace(r"/", r"-")
=======
def create_data_frame(arr, transformer, index):
    return pd.DataFrame(
        arr,
        columns=transformer.get_feature_names_out() if transformer else None,
        index=index,
    )


def read_csv(path, *args, **kwargs):
    data = pd.read_csv(path)
    data.columns = (
        data.columns.str.replace(r" ", r"_")
        .str.replace(r"(", r"")
        .str.replace(r")", r"")
        .str.replace(r"/", r"-")
>>>>>>> 50d5f19 (Plots and scoring working.)
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

<<<<<<< HEAD
model = models.best_estimator_
pred = model.predict(test_scaled)
pred_prob = model.predict_proba(test_scaled)
||||||| parent of 50d5f19 (Plots and scoring working.)

def main(path="../Data/Training/"):
    files = glob.glob(rf"{path}**/*.csv", recursive=True)

    def read_gen(files):
        for file in files:
            yield read_csv(file, pipe=pipe[:-1])

    for file, data in zip(files, read_gen(files)):
        scatterplot(data, file.replace(r"csv", r"png"))


if __name__ == "__main__":
    main()
=======

def grid_search(
    X,
    y,
    param_grid={
        "C": np.exp(np.linspace(np.log(1 / 10), 0, 10)),
        "l1_ratio": np.exp(np.linspace(np.log(1 / 10), 0, 10)),
    },
    model=LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        n_jobs=-1,
        multi_class="ovr",
    ),
    max_iter=500,
    verbose=False,
    return_train_score=False,
):
    model.set_params(max_iter=max_iter)
    model = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=return_train_score,
    ).fit(X, y)
    return model


def build_training(path):
    """
    Compile all training datasets in path and concatenate into one dataframe.

    path (str): The directory containing training sets.
    """
    datasets = glob.iglob(os.path.join(path, r"**/*.csv"), recursive=True)
    dataset_list = []
    for dataset in datasets:
        data = pd.read_csv(dataset, index_col="UUID")
        dataset_list.append(data)
    print(f"Training on {len(dataset_list)} datasets.")
    df = pd.concat(dataset_list, axis=0)
    df = format_columns(df)
    return df


def format_columns(data):
    """
    Format column strings.

    data (pd.DataFrame): The data whose columns are to be formatted.
    """
    copy = data.copy()
    copy.columns = (
        data.columns.str.replace(" ", "_", regex=False)
        .str.replace("/", "-", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.lower()
    )
    return copy


def main(path="../Data/Training/"):
    files = glob.glob(rf"{path}**/*.csv", recursive=True)

    def read_gen(files):
        for file in files:
            yield read_csv(file, pipe=pipe[:-1])

    for file, data in zip(files, read_gen(files)):
        scatterplot(data, file.replace(r"csv", r"png"))
        yield data

<<<<<<< Updated upstream

||||||| constructed merge base
<<<<<<< Updated upstream
||||||| constructed merge base
breakpoint()
training_scaled = pipe.fit_transform(data[0])
test_scaled = pipe.transform(data[1])
=======
training_scaled = pipe.fit_transform(data[0])
test_scaled = pipe.transform(data[1])
>>>>>>> Stashed changes

=======
>>>>>>> Stashed changes
if __name__ == "__main__":
    data_gen = main()


    data = build_training(rf"../Data/Training/OperatorPlusDefault/")
    pca = PCA()
    pipe = Pipeline(
        [
            ("scaler", MinMaxScaler((0.0001, 1))),
            ("poly", PolynomialFeatures()),
        ]
    )

    points = np.log(
        pipe.fit_transform(data.drop(["operator_classification", "class"], axis=1))
    )
    models_0 = grid_search(
        points,
        data["operator_classification"],
        param_grid={
            "C": np.exp(np.linspace(np.log(1 / 5), 0, 5)),
            "l1_ratio": np.exp(np.linspace(np.log(1 / 5), 0, 5)),
        },
        verbose=3,
    )
    print(f"Untransformed: {models_0.best_score_}")

    pca_points = pca.fit_transform(points)
    models_pca = grid_search(
        pca_points,
        data["operator_classification"],
        param_grid={
            "C": np.exp(np.linspace(np.log(1 / 5), 0, 5)),
            "l1_ratio": np.exp(np.linspace(np.log(1 / 5), 0, 5)),
        },
        verbose=3,
    )
    print(f"PCA: {models_pca.best_score_}")

    dummies = pd.get_dummies(data["operator_classification"])
    alg = ph.create(dummies)
    points_resid = alg.residualize(points)
    models_resid = grid_search(
        points_resid,
        data["class"],
        param_grid={
            "C": np.exp(np.linspace(np.log(1 / 5), 0, 5)),
            "l1_ratio": np.exp(np.linspace(np.log(1 / 5), 0, 5)),
        },
        verbose=3,
    )
    print(f"Fixed Effects: {models_resid.best_score_}")


    x = points_resid[:, 0]
    y = points_resid[:, 1]
>>>>>>> 50d5f19 (Plots and scoring working.)
