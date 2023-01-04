#!/usr/bin/env python3
# Copyright Neko Juers 2022

from datetime import date
import glob
from pathlib import Path
import os
from pickle import load, dump
import platform
import re

import click
import numpy as np
import pandas as pd
import pyhdfe as ph
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.utils import check_X_y


def reduce(data, pca, threshold=0.95):
    var_comp = pca.explained_variance_ratio_
    cum_var = []
    i_var = []
    for i, var in var_comp:
        var = var + var_comp[i + 1]
        if var <= threshold:
            cum_var.append(var)
            i_var.append(i)
        return data.loc[:, data.columns[i_var]]


class Residualizer(BaseEstimator):
    """A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        self.feature_names_in_ = X.columns
        X, y = check_X_y(X, y)
        self.classes = y
        self.dummies = pd.get_dummies(self.classes)
        self.transformer = ph.create(self.dummies)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def get_feature_names_out(self, input_features=None):
        feature_names_out = self.feature_names_in_
        return feature_names_out

    def transform(self, X):
        out = self.transformer.residualize(X)
        return out

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    # def predict(self, X):
    #     """ A reference implementation of a predicting function.
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape (n_samples, n_features)
    #         The training input samples.
    #     Returns
    #     -------
    #     y : ndarray, shape (n_samples,)
    #         Returns an array of ones.
    #     """
    #     X = check_array(X, accept_sparse=True)
    #     check_is_fitted(self, 'is_fitted_')
    #     return np.ones(X.shape[0], dtype=np.int64)


class ModelOutput:
    def __init__(self, model, pipe, training):
        self.models = model
        self.pipe = pipe
        self.training = training
        return None


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


def generate_param_grid(C=(0, 1, 5), l1_ratio=(0, 1, 5)):
    # Insert is used to add the zero value, which cannot be added through log
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


def newest(path):
    """
    Given path, find most recent file.
    """
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


# def remove_pcs(data, model, threshold=0.95):
#     var = model.named_steps["pca"].explained_variance_ratio_
#     cum_var = []
#     for i, _ in enumerate(var):
#         cum_var.append(var[: i + 1].sum())
#     cum_var_bool = np.array(cum_var) < threshold
#     return data.iloc[:, cum_var_bool]


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
@click.option(
    "--suffix",
    "-su",
    default=None,
    type=str,
    help="""
        The string to append to output files.
    """,
)
@click.pass_context
def cli(ctx, polynomial_degree, ignore_unknown, training_set, suffix):
    # Combines all csvs in training directory into a single dataframe
    ctx.obj["suffix"] = suffix if suffix else ""
    ctx.obj["data"] = build_training(training_set)
    if ignore_unknown:
        training = (
            ctx.obj["data"].copy().loc[ctx.obj["data"]["Class"] != "Unknown"]
        )
    else:
        training = ctx.obj["data"].copy()
    ctx.obj["unknown_str"] = "_ignore" if ignore_unknown else ""
    ctx.obj["polynomial_degree"] = polynomial_degree
    # Isolate response data.
    try:
        ctx.obj["training_response"] = training["operator_classification"]
        ctx.obj["default_response"] = training["class"]
    except KeyError:
        ctx.obj["training_response"] = training["class"]
        ctx.obj["default_response"] = np.full(training.shape[0], -1)
    ctx.obj["pipe"] = Pipeline(
        [
            ("scaler", MinMaxScaler()),
        ]
    )
    try:
        training.drop("operator_classification")
    except KeyError:
        pass
    training.drop(
        ["class", "ch2-ch1_ratio", "aspect_ratio"], axis=1, inplace=True
    )
    training = training.select_dtypes(float)
    ctx.obj["training_columns"] = training.columns
    training_arr = pd.DataFrame(
        Residualizer().fit_transform(training, ctx.obj["default_response"]),
        columns=training.columns,
        index=training.index,
    )
    training_scaled = pd.DataFrame(
        ctx.obj["pipe"].fit_transform(
            training_arr, ctx.obj["training_response"]
        ),
        columns=ctx.obj["training_columns"],
        index=training.index,
    )
    ctx.obj["training"] = training_scaled
    # ctx.obj["training"] = remove_pcs(
    #     training_scaled, ctx.obj["pipe"]
    # )


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
    default=1,
    required=False,
    type=click.IntRange(0, 3),
    help="""
        Controls the verbosity of the grid search cross-validation.
        From sklearn documentation:
            1 : the computation time for each fold and parameter candidate is displayed;
            2 : the score is also displayed;
            3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
    """,
)
@click.option(
    "--return-train-score/--no-return-train-score",
    "-r",
    type=bool,
    default=False,
    help="""
        Include training score in results. Can increase computation time.
    """,
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
        verbose=verbose,
        return_train_score=return_train_score,
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
    print(
        f"Best params: {models.best_params_}, Best score: {models.best_score_}"
    )
    out = ModelOutput(models, ctx.obj["pipe"], ctx.obj["data"].reset_index())
    with open(
        (
            f"../Data/Models/{date.today()}"
            f"_C-{c_str}_l1-{l1_str}"
            f"_p-{ctx.obj['polynomial_degree']}"
            f"{ctx.obj['unknown_str']}{ctx.obj['suffix']}.pickle"
        ),
        "wb",
    ) as file:
        dump(out, file)
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
    model = models.models.best_estimator_
    # Data cleaning and preprocessing.
    for file in unclassified_path:
        print(f"processing {file}")
        # Output filename manipulation
        basename = os.path.basename(file)
        output_base = re.sub(
            ".csv", f"{ctx.obj['suffix']}_classified.csv", basename
        )
        outfile = rf"../Data/Classified/{output_base}"
        # Unclassified data formatting
        data = pd.read_csv(file, index_col="UUID")
        unclassified = format_columns(data)
        data.reset_index(inplace=True)
        capture_id = unclassified["capture_id"]
        try:
            default_classification = unclassified["class"]
        except KeyError:
            default_classification = np.full(unclassified.shape[0], -1)
        unclassified = unclassified.loc[:, ctx.obj["training_columns"]]
        unclassified = pd.DataFrame(
            Residualizer().fit_transform(unclassified, default_classification),
            columns=unclassified.columns,
            index=unclassified.index,
        )
        unclassified_scaled = pd.DataFrame(
            models.pipe.transform(unclassified),
            columns=models.pipe.get_feature_names_out(),
            index=data.index,
        )
        # Remove pcs. explaining minority of variance
        # unclassified_scaled = remove_pcs(
        #     unclassified_scaled, ctx.obj["pipe"]
        # )
        # Import most recent fitted models and best estimator
        # Fucking Windows... :/
        try:
            if "Windows" in platform.platform():
                windows_path = Path(r"..\Data\Models")
                list_of_files = list(
                    windows_path.glob(rf"*{ctx.obj['unknown_str']}.pickle")
                )
            else:
                list_of_files = glob.glob(
                    f"../Data/Models/*{ctx.obj['unknown_str']}.pickle"
                )
            latest_file = max(list_of_files, key=os.path.getctime)
        except ValueError:
            print("No model file matching ignore flag.")
            return None
        with open(latest_file, "rb") as file:
            models = load(file)
        print(
            f"File: {unclassified_path}, best params: {models.models.best_params_}, best score: {models.models.best_score_}"
        )
        model = models.models.best_estimator_
        # Generate predictions
        predicted = model.predict(unclassified_scaled)
        data["Class"] = predicted
        data.to_csv(outfile)
    return None


if __name__ == "__main__":
    cli(obj={})
