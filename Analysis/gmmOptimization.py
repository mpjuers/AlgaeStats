#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import functools as ft
import itertools as it
import os
import pandas as pd
from pickle import dump, load
import math

import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
from scipy.stats import gmean
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def fit_gmm(
    combination, n_components, data_training, data_test, *args, **kwargs
):
    print(f"fitting {str(combination)} with {n_components} components.")
    try:
        model = GaussianMixture(n_components=n_components).fit(
            data_training[list(combination)]
        )
        predict = model.predict(data_test[list(combination)])
        max_correct = 0.0
        test_class = data_test["class"].astype("category").cat.codes
        data_test = data_test.drop(["class"], axis=1)
        # Figure out component assignments.
        for i in range(n_components):
            correct = (predict == i) == test_class
            percent_correct = correct.sum() / data_test.shape[0]
            if percent_correct > max_correct:
                max_correct = percent_correct
        likelihood = model.score(data_test[list(combination)])
        print(
            f"{str(combination)} -- percent correct: {max_correct}, likelihood: {model.score(data_test[list(combination)])}"
        )
        return [
            *[str(x) for x in combination],
            n_components,
            likelihood,
            max_correct,
        ]
    except ValueError:
        return [
            *[str(x) for x in combination],
            n_components,
            float("NaN"),
            float("NaN"),
        ]


def optimize_gmm(
    data,
    data_test,
    combinations=None,
    n_iter=6,
    correct_threshold=0.9,
    percent_correct=0.5,
    max_components=3,
    anneal=0.5,
    max_models=1000,
    # Below only used internally
    models={},
    p=0.0,
    n=0,
    pool=mp.Pool(mp.cpu_count()),
    *args,
    **kwargs,
):
    print(f"correct threshold set to {percent_correct}")
    print(f"max percent correct == {p}")
    # Evaluate recursion conditions.
    if p < correct_threshold:
        print(f"p == {p}, correct_threshold == {correct_threshold}")
        if n < n_iter:
            print(f"n == {n}, n_iter == {n_iter}")
            # Initialize with existing columns
            if combinations is None:
                new_combinations = [[x] for x in data.columns]
            else:
                # Error handling for single objects rather than iterables
                if n > 1:
                    new_combinations = list(
                        list(it.chain(x, [y]))
                        for x, y in it.product(combinations, data.columns)
                        if y not in x
                    )
                else:
                    new_combinations = list(
                        list(it.chain([x], [y]))
                        for x, y in it.product(combinations, data.columns)
                        if y not in x
                    )
            # Do the actual work.
            new_combinations = list(
                it.product(new_combinations, range(2, max_components + 1))
            )
            # Remove duplicates
            if n > 0:
                new_combinations = set((tuple(set(x[0])), x[1]) for x in new_combinations)
            print(f"Fitting {len(new_combinations)} models with new input.")
            models[n] = pd.DataFrame(
                pool.starmap(
                    ft.partial(
                        fit_gmm,
                        data_training=data,
                        data_test=data_test,
                    ),
                    new_combinations,
                ),
            ).dropna()
            models[n].columns = it.chain(
                range(n + 1),
                ["n_components"],
                ["likelihood"],
                ["percent_correct"],
            )
            models[n]["rank"] = models[n]["percent_correct"].rank(pct=True)
            # Get rid of unexplanatory results according to correct threshold.
            models[n] = (
                models[n]
                .sort_values("rank", ascending=False)
                .loc[(models[n]["percent_correct"] > percent_correct), :]
            )
            if models[n].shape[0] == 0:
                print("Out of models")
                pool.terminate()
                return models
            # Keep number of models managable by reducing input combinations
            elif models[n].shape[0] * len(data.columns) >= max_models:
                models[n] = models[n].iloc[
                    : math.floor(
                        max_models / (max_components - 1) / len(data.columns)
                    ),
                    :,
                ]
            print(f"Done fitting iteration {n}.")
            # Set multiindex to model combinations.
            models[n].set_index(
                models[n].columns[:-4].tolist(), inplace=True, drop=True
            )
            # Summary
            # Average correct
            p = models[n]["percent_correct"].max()
            summary_n = models[n]["percent_correct"].agg(
                ["mean", "std", "median", "max", "min"]
            )
            print(f"selected {models[n].shape[0]} feature combinations")
            print(models[n])
            print(summary_n)
            if n > 0:
                summary_n_minus = models[n - 1]["percent_correct"].agg(
                    ["mean", "std", "median", "max"]
                )
                print(pd.DataFrame([summary_n_minus, summary_n]))
                print(
                    f"delta percent correct: {summary_n['mean'] - summary_n_minus['mean']}, {(summary_n['mean'] - summary_n_minus['mean']) / summary_n_minus['mean'] * 100}%"
                )
                if summary_n.loc["max"] - summary_n_minus.loc["max"] < 0.001:
                    print(
                        f"delta percent_correct insufficient to iterate further."
                    )
                    return models
            # Call optimize_gmm with updataed parameters
            return optimize_gmm(
                data,
                data_test,
                models=models,
                combinations=models[n].index,
                n_iter=n_iter,
                correct_threshold=correct_threshold,
                n=n + 1,
                p=p,
                anneal=anneal,
                percent_correct=percent_correct
                + (1 - percent_correct) * anneal,
                max_components=max_components,
                pool=pool,
            )
        else:
            print(f"Model failed to converge after {n_iter} iterations.")
    elif p is float("NaN"):
        print("Out of models.")
    else:
        print("Model converged.")
    pool.terminate()
    return (models,)


if __name__ == "__main__":
    # Data import and cleaning
    training = pd.read_csv("../Data/bigOlDF.csv")
    test = pd.read_csv("../Data/2022-07-26_test4_with-classification.csv")
    test.columns = (
        test.columns.str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    test_class = test["class"] == ("Cyanobacteria" or 1)
    test_clean = test.drop(
        [
            "class",
            "capture_id",
            "capture_x",
            "capture_y",
            "elapsed_time",
            "image_height",
            "image_width",
            "uuid",
            "group_id",
            "calibration_factor",
        ],
        axis=1,
    )
    test_clean.insert(test_clean.shape[1], "class", test_class)
    training.drop(
        set(training.columns) - set(test_clean.columns), axis=1, inplace=True
    )

    commit = os.popen("git rev-parse HEAD").read().strip()
    try:
        with open(f"../Data/{commit}.pickle", "rb") as file:
            models = load(file)
    except FileNotFoundError:
        models = optimize_gmm(
            training.iloc[:, :10],
            test,
            anneal=0.25,
            percent_correct=0.55,
            max_models=3000,
            max_components=5,
        )
        with open(f"../Data/{commit}.pickle", "wb") as file:
            dump(models, file)
    breakpoint()
