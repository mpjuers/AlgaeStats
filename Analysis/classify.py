#!/usr/bin/env python3
# Copyright Neko Juers 2022

from datetime import date
import glob
import os
from pickle import load, dump
import re

import click
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
        penalty="elasticnet",
        solver="saga",
        n_jobs=-1,
        multi_class="ovr",
    ),
    max_iter=100,
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


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def remove_pcs(data, model, threshold=0.95):
    var = model.explained_variance_ratio_
    cum_var = []
    for i, _ in enumerate(var):
        cum_var.append(var[: i + 1].sum())
    cum_var_bool = np.array(cum_var) < threshold
    return data.iloc[:, cum_var_bool]


@click.group(chain=True)
@click.option(
    "--polynomial-degree",
    "-p",
    default=1,
    required=False,
    help="""
       How many interaction terms to include. 
    """,
)
@click.option(
    "--ignore-unknown/--keep-unknown",
    "-i",
    default=False,
    required=False,
    help="""
        Whether to keep unknown datapoints.
    """,
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
@click.pass_context
def cli(ctx, polynomial_degree, ignore_unknown, training_set):
    # Combines all csvs in training directory into a single dataframe
    data = build_training(training_set)
    if ignore_unknown:
        training = data.copy().loc[data["class"] != "Unknown"]
    else:
        training = data.copy()
    ctx.obj["unknown_str"] = "_ignore" if ignore_unknown else ""
    ctx.obj["polynomial_degree"] = polynomial_degree
    ctx.obj["pipe"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=polynomial_degree)),
            ("pca", PCA()),
        ]
    )
    # Isolate response data.
    ctx.obj["training_response"] = training["class"]
    training.drop(
        ["class", "ch2-ch1_ratio", "aspect_ratio"], axis=1, inplace=True
    )
    training = training.select_dtypes(float)
    training_scaled = pd.DataFrame(
        ctx.obj["pipe"].fit_transform(training),
        columns=ctx.obj["pipe"].get_feature_names_out(),
        index=training.index,
    )
    ctx.obj["training"] = remove_pcs(
        training_scaled, ctx.obj["pipe"].named_steps["pca"]
    )


@cli.command("train")
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
    "--max-iter",
    "-m",
    default=1000,
    required=False,
    type=int,
    help="""
        The maximum number of iterations used to fit the logistic regression.
    """,
)
@click.option(
    "--verbose",
    "-v",
    default=0,
    required=False,
    type=click.IntRange(0, 3),
    help="""
        Controls the verbosity of the grid search cross-validation.
        From sklearn documentation:
            1 : the computation time for each fold and parameter candidate is displayed;
            2 : the score is also displayed;
            3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
    """
)
@click.option(
    "--return-train-score/--no-return-train-score",
    "-r",
    type=bool,
    default=False,
    help="""
        Include training score in results. Can increase computation time.
    """
)
@click.pass_context
def train(
    ctx,
    c_grid,
    l1_grid,
    max_iter,
    verbose,
    return_train_score,
):
    # Generate combinations of C and l1_ratio for model training.
    # Fit models.
    models = grid_search(
        ctx.obj["training"],
        ctx.obj["training_response"],
        param_grid=generate_param_grid(C=c_grid, l1_ratio=l1_grid),
        max_iter=max_iter,
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
    with open(
        f"../Data/Models/{date.today()}_C-{c_str}_l1-{l1_str}_p-{ctx.obj['polynomial_degree']}{ctx.obj['unknown_str']}.pickle",
        "wb",
    ) as file:
        dump(models, file)
    return None


@cli.command("classify")
@click.argument(
    "unclassified_path",
    nargs=-1,
    required=False,
)
@click.option(
    "--ignore-unknown/--keep-unknown",
    "-i",
    default=False,
    required=False,
    help="""
        Whether to keep unknown datapoints.
    """,
)
@click.pass_context
def classify(
    ctx,
    unclassified_path,
    ignore_unknown,
):
    with open(newest("../Data/Models"), "rb") as file:
        models = load(file)
    # Extract model with highest likelihood.
    model = models.best_estimator_
    # Data cleaning and preprocessing.
    for file in unclassified_path:
        print(f"processing {file}")
        basename = os.path.basename(file)
        output_base = re.sub(".csv", "_classified.csv", basename)
        outfile = f"../Data/Classified/{output_base}"
        unclassified = format_columns(pd.read_csv(file, index_col="UUID"))
        capture_id = unclassified["capture_id"]
        unclassified = unclassified.loc[
            :, ctx.obj["training"].columns.intersection(unclassified.columns)
        ]
        unclassified_scaled = pd.DataFrame(
            ctx.obj["pipe"].transform(unclassified),
            columns=ctx.obj["pipe"].get_feature_names_out(),
            index=unclassified.index,
        )
        unclassified_scaled = remove_pcs(
            unclassified_scaled, ctx.obj["pipe"].named_steps["pca"]
        )
        # Import fitted models
        list_of_files = glob.glob(
            f"../Data/Models/*{ctx.obj['unknown_str']}.pickle"
        )  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, "rb") as file:
            models = load(file)
        print(
            f"File: {unclassified_path}, best params: {models.best_params_}, best score: {models.best_score_}"
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
    cli(obj={})
