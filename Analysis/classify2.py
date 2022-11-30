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
from sklearn.linear_model import LogisticRegression
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
        predict = LogisticRegression(penalty="none").fit(X_resample, y_resample).predict(X_resample)
        return predict
    with Pool(os.cpu_count()) as pool:
        results = pool.starmap_async(fit_model(X, y, model), (() for _ in range(n)))
        results.wait()
        results = np.array(results.get())
        pool.terminate()
    return np.apply_along_axis(lambda x: x.sum() / len(x), 1, results)


def grid_search(params, X, X_test, y, y_test, *args, **kwargs):
    params = list(params)
    with Pool(os.cpu_count()) as pool:
        models = pool.map_async(
            lambda x: LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=x[0],
                l1_ratio=x[1],
                max_iter=100,
            ),
            params,
            chunksize=len(params) // os.cpu_count(),
        )
        models.wait()
        pool.terminate()
    out = np.empty((len(params), 5), dtype=float)
    models_out = []
    for i, model in enumerate(models.get()):
        model.fit(X, y)
        predict = model.predict(X_test)
        # model_selected_features = SelectFromModel(estimator=model)
        # features_to_keep = model_selected_features.get_support()
        # model_test_features_kept = X_test.loc[:, features_to_keep]
        # predict = (
        #     LogisticRegression(penalty="none")
        #     .fit(model_test_features_kept, y_test)
        #     .predict(model_test_features_kept)
        # )
        out[i, 0] = params[i][0]
        out[i, 1] = params[i][1]
        out[i, 2] = (y_test == predict).sum() / X_test.shape[0]
        out[i, 3] = model.score(X_test, y_test)
        out[i, 4] = np.nan
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
    out = out.sort_values("likelihood", ascending=False)
    print("Summary of fit models:")
    print(out)
    return out


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
    "--grid-partitions",
    "-g",
    type=int,
    default=10,
    required=False,
    help="""
        The number of grid partitions on each C and l1_ratio dimension.
        The grid search will process n by n models.
        Default: 10
    """,
)
def main(unclassified_path, train, training_set, grid_partitions):
    # Combines all csvs in training directory into a single dataframe
    data = build_training(training_set)
    # Isolate response data.
    response = data["class"]
    data = data.select_dtypes(float)
    data.drop(["ch2-ch1_ratio", "aspect_ratio"], axis=1, inplace=True)
    training = data.sample(frac=0.5)
    response_training = response.loc[training.index]
    scaler = StandardScaler()
    training_scaled = pd.DataFrame(
        scaler.fit_transform(training),
        columns=training.columns,
        index=training.index,
    )
    test = data.drop(training.index)
    test_scaled = pd.DataFrame(
        scaler.transform(test),
        columns=test.columns,
        index=test.index,
    )
    response_test = response.loc[test_scaled.index]
    if train:
        # Generate combinations of C and l1_ratio for model training.
        grid = it.product(
            np.exp(
                np.linspace(math.log(1 / grid_partitions), 0, grid_partitions)
            ),
            np.exp(
                np.linspace(math.log(1 / grid_partitions), 0, grid_partitions)
            ),
        )
        # Fit models.
        models = grid_search(
            grid,
            training_scaled,
            test_scaled,
            response_training,
            response_test,
        )
        with open(
            f"../Data/Models/{date.today()}_{grid_partitions}.pickle", "wb"
        ) as file:
            dump(models, file)
    else:
        with open(newest("../Data/Models"), "rb") as file:
            models = load(file)
    # Extract model with highest likelihood.
    model_details = models.loc[models["likelihood"].idxmax()]
    print(
        f"Best model parameters: C={model_details['C']}, l1_ratio={model_details['l1_ratio']}"
    )
    model = model_details["model"]

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
                scaler.transform(unclassified), columns=unclassified.columns
            )
            # Import fitted models
            list_of_files = glob.glob("../Data/Models/*.pickle") # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            with open(latest_file, "rb") as file:
                models = load(file)
                model_details = models.loc[models["likelihood"].idxmax()]
                breakpoint()
                print(
                    f"C={model_details['C']}, l1_ratio={model_details['l1_ratio']}"
                )
                model = model_details["model"]
            # Generate predictions
            predicted = model.predict(unclassified_scaled)
            # predicted = predict(
            #     training_scaled, response_training, unclassified, model
            # )
            # Return unscaled input data with predictions.
            classified = unclassified
            unclassified["class"] = predicted
            classified["capture_id"] = capture_id
            classified.to_csv(outfile)
    return None


if __name__ == "__main__":
    main()
