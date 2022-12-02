#!/usr/bin/env python3
# Copyright Neko Juers 2022

from datetime import date
import glob
import itertools as it
import math
from multiprocess import Pool
import os
from pickle import load, dump
import re

import click
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


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
    df = pd.concat(dataset_list, axis=0)
    df = format_columns(df)
    return df


def format_columns(data):
    """
    Format column strings.

    data (pd.DataFrame): The data whose columns are to be formatted.
    """
    data.columns = (
        data.columns.str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    return data


def predict(training, response, test, model):
    """
    Generate prediction from trained model.

    training (pd.DataFrame): The data used to fit the model.
    response (pd.Series): The categorical response data for the training set.
    test (pd.DataFrame): The data for which responses are to be predicted.
    model (LogisticRegression): The model used to fit predicted response data.
    """
    features = training.columns[SelectFromModel(model).get_support()]
    training = training.loc[:, features]
    test = test.loc[:, features]
    response = model.fit(training, response).predict(test)
    return response


def cross_validate(X, y, model, n=100):
    def fit_model(X, y, model):
        X_resample = X.resample()
        y_resample = y.resample()
        model_selected_features = SelectFromModel(estimator=model)
        features_to_keep = model_selected_features.get_support()
        X_resample = X_resample.loc[:, features_to_keep]
        predict = (
            LogisticRegression(penalty="none")
            .fit(X_resample, y_resample)
            .predict(X_resample)
        )
        return predict

    with Pool(os.cpu_count()) as pool:
        results = pool.starmap_async(
            fit_model(X, y, model), (() for _ in range(n))
        )
        results.wait()
        results = np.array(results.get())
        pool.terminate()
    return np.apply_along_axis(lambda x: x.sum() / len(x), 1, results)


def generate_param_grid(C=(0, 1, 5), l1_ratio=(0, 1, 5)):
    out = {
        "C": np.exp(
            np.linspace(
                np.log(C[0] + (C[1] - C[0]) / C[2]), np.log(C[1]), C[2]
            )
        ),
        "l1_ratio": np.exp(
            np.linspace(
                np.log(
                    l1_ratio[0] + (l1_ratio[1] - l1_ratio[0]) / l1_ratio[2]
                ),
                np.log(l1_ratio[1]),
                C[2],
            )
        ),
    }
    return out


def grid_search(
    X,
    y,
    param_grid={
        "C": np.exp(np.linspace(np.log(1 / 10), 0, 10)),
        "l1_ratio": np.exp(np.linspace(np.log(1 / 10), 0, 10)),
    },
    model=LogisticRegression(
        penalty="elasticnet", solver="saga", n_jobs=-1, multi_class="ovr"
    ),
):
    model = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
    ).fit(X, y)
    return model


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


@click.command()
@click.argument(
    "unclassified_path",
    nargs=-1,
    required=False,
)
@click.option(
    "--train/--no-train",
    "-t",
    type=bool,
    default=False,
    required=True,
    help="Whether to train the model and generate grid search.",
)
@click.option(
    "--training-set",
    "-s",
    type=str,
    default="../Data/Training",
    required=False,
    help="""
        The directory containing training datasets.
        These will be concatenated into a single dataset.
        Default: ../Data/Training
    """,
)
@click.option(
    "--C-grid",
    "-C",
    default=(0, 1, 10),
    required=False,
    type=(float, float, int),
    help="""
        The min, max and number of partitions for the C grid search.
    """,
)
@click.option(
    "--l1-grid",
    "-l1",
    default=(0, 1, 10),
    required=False,
    type=(float, float, int),
    help="""
        The min, max and number of partitions for the l1 grid search.
    """,
)
@click.option(
    "--ignore-unknown/--keep-unknown",
    "-i",
    default=False,
    required=False,
    help="""
        Whether to keep unknown datapoints.
    """
)
def main(unclassified_path, train, training_set, c_grid, l1_grid, ignore_unknown):
    pipe = Pipeline(
        [("poly", PolynomialFeatures()), ("scaler", StandardScaler())]
    )
    # Combines all csvs in training directory into a single dataframe
    data = build_training(training_set)
    if ignore_unknown:
        data = data.loc[data["class"] != "Unknown"]
    # Isolate response data.
    response = data["class"]
    data = data.select_dtypes(float)
    data.drop(["ch2-ch1_ratio", "aspect_ratio"], axis=1, inplace=True)
    training = data
    training_response = response.loc[training.index]
    training_scaled = pd.DataFrame(
        pipe.fit_transform(training),
        columns=pipe.get_feature_names_out(),
        index=training.index,
    )
    if train:
        # Generate combinations of C and l1_ratio for model training.
        # Fit models.
        models = grid_search(
            training_scaled,
            training_response,
            param_grid=generate_param_grid(C=c_grid, l1_ratio=l1_grid),
        )
        c_str = (
            str(c_grid)
            .replace(",", "-")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
        )
        l1_str = (
            str(l1_grid)
            .replace(",", "-")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
        )
        unknown_str = "_ignore" if ignore_unknown else ""
        with open(
            f"../Data/Models/{date.today()}_C-{c_str}_l1-{l1_str}{unknown_str}.pickle", "wb"
        ) as file:
            dump(models, file)
    else:
        with open(newest("../Data/Models"), "rb") as file:
            models = load(file)
    # Extract model with highest likelihood.
    model = models.best_estimator_

    # Data cleaning and preprocessing.
    if unclassified_path:
        for file in unclassified_path:
            print(f"processing {file}")
            basename = os.path.basename(file)
            output_base = re.sub(".csv", "_classified.csv", basename)
            outfile = f"../Data/Classified/{output_base}"
            unclassified = format_columns(pd.read_csv(file, index_col="UUID"))
            capture_id = unclassified["capture_id"]
            unclassified = unclassified.loc[
                :, training.columns.intersection(unclassified.columns)
            ]
            unclassified_scaled = pd.DataFrame(
                pipe.transform(unclassified),
                columns=pipe.get_feature_names_out(),
                index=unclassified.index,
            )
            # Import fitted models
            list_of_files = glob.glob(
                f"../Data/Models/*{unknown_str}.pickle"
            )  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            with open(latest_file, "rb") as file:
                models = load(file)
                print(
                    f"best params: {models.best_params_}, best score: {models.best_score_}"
                )
                model = models.best_estimator_
            # Generate predictions
            predicted = model.predict(unclassified_scaled)
            classified = unclassified
            unclassified["class"] = predicted
            classified["capture_id"] = capture_id
            classified.to_csv(outfile)
    return None


if __name__ == "__main__":
    main()
