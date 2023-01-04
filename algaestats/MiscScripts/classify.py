#!/usr/bin/env python3

import click
from datetime import date
from glob import iglob
import itertools as it
import math
import os
from pickle import load, dump
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def split_data(data):
    training = data.sample(frac=0.5)
    test = data.drop(training.index)
    return {"training": training, "test": test}


def grid_search(params, X, X_test, y, y_test, *args, **kwargs):
    models = [
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=x[0],
            l1_ratio=x[1],
            max_iter=100,
        )
        for x in params
    ]
    out = np.empty((len(params), 5), dtype=float)
    models_out = []
    for i, model in enumerate(models):
        model.fit(X, y)
        model_selected_features = SelectFromModel(estimator=model)
        features_to_keep = model_selected_features.get_support()
        model_test_features_kept = X_test.loc[:, features_to_keep]
        predict = (
            LogisticRegression(penalty="none")
            .fit(model_test_features_kept, y_test)
            .predict(model_test_features_kept)
        )
        out[i, 0] = params[i][0]
        out[i, 1] = params[i][1]
        out[i, 2] = (y_test == predict).sum() / X_test.shape[0]
        out[i, 3] = model.score(X_test, y_test)
        out[i, 4] = features_to_keep.sum()
        models_out.append(model)
    out = pd.DataFrame(
        out,
        columns=[
            "C",
            "l1_ratio",
            "percent_correct",
            "likelihood",
            "no_features_kept",
        ],
    )
    out["model"] = models_out
    out = out.sort_values("percent_correct", ascending=False)
    print(out)
    return out


def build_training(path):
    datasets = iglob(os.path.join(path, "*.csv"))
    dataset_list = []
    for dataset in datasets:
        data = pd.read_csv(dataset, index_col="UUID")
        dataset_list.append(data)
    df = pd.concat(dataset_list, axis=0)
    df = format_columns(
        df,
        drop_columns=[
            "calibration_factor",
            "capture_id",
            "capture_x",
            "capture_y",
            "group_id",
            "image_height",
            "image_width",
            "elapsed_time",
            "ch2-ch1_ratio",
            "aspect_ratio",
        ],
    )
    return df


def train_model(data, grid_partitions=10, *args, **kwargs):
    # drop uninformative columns
    response = data["class"]
    data.drop("class", axis=1, inplace=True)
    data_dict = split_data(data)
    response_dict = {
        "training": response.loc[data_dict["training"].index],
        "test": response.loc[data_dict["test"].index],
    }
    # generate grid search
    grid = [
        [math.exp(x), math.exp(y)]
        for x, y in (
            it.product(
                np.linspace(
                    math.log(1 / kwargs["grid_partitions"]),
                    math.log(1.0),
                    kwargs["grid_partitions"],
                ),
                np.linspace(
                    math.log(1 / kwargs["grid_partitions"]),
                    math.log(1.0),
                    kwargs["grid_partitions"],
                ),
            )
        )
    ]
    grid_results = grid_search(
        grid,
        data_dict["training"],
        data_dict["test"],
        response_dict["training"],
        response_dict["test"],
    )
    return grid_results


def format_columns(data, drop_columns):
    data.columns = (
        data.columns.str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    return data.drop(drop_columns, axis=1, inplace=False)


@click.command()
@click.option("--train/--no-train", default=False)
@click.option(
    "--training_set",
    "-t",
    type=str,
    required=False,
    default="../Data/Training/",
)
@click.option(
    "--models",
    "-m",
    required=False,
    default="ls -Art ../Data/Models/*.pickle | tail -n 1",
    type=str,
)
@click.option(
    "--unidentified",
    "-u",
    type=str,
    required=False,
)
@click.option(
    "--out", "-o", default=f"classified_{date.today()}.csv", type=str
)
def main(train, unidentified, models, out, training_set, *args, **kwargs):
    training = build_training(training_set).select_dtypes(float)
    if train:
        pass
    else:
        with open(os.popen(models).read().strip(), "rb") as model_file:
            models = load(model_file)
            print(models)
    best_model = models["model"].loc[models["likelihood"].idxmax()]
    features_to_keep = training.columns[SelectFromModel(best_model).get_support()]
    test = pd.read_csv(unidentified).iloc[:, features_to_keep]
    scaler = StandardScaler().fit(training)
    training = pd.DataFrame(scaler.transform(training), columns=training.columns[features_to_keep])
    test = pd.DataFrame(scaler.transform(test), columns=test.columns[features_to_keep])


if __name__ == "__main__":
    main()
