#!/use/bin/env python3
# Copyright 2022 Neko Juers

import itertools as it
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.utils import resample


def format_columns(data, drop_columns):
    data.columns = (
        data.columns.str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    return data.drop(drop_columns, axis=1, inplace=False)


def load_data(path):
    out = pd.read_csv(path)
    return out


def grid_search(params, X, X_test, y, y_test):
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
    out = out.sort_values("no_features_kept")
    print(out)
    return out


def scatterplot(data, response, model):
    selected = SelectFromModel(estimator=model)
    keep = selected.get_support()
    data = data.iloc[:, keep]
    predict = (
        LogisticRegression(penalty="none")
        .fit(data, response)
        .predict(data)
        .astype(int)
    )
    correct = predict == response.values
    data["predict"] = predict
    data["correct"] = correct
    data["response"] = response.values
    sns.scatterplot(
        data,
        x="ch2-ch1_ratio",
        y="aspect_ratio",
        hue="response",
        style="correct",
    )
    plt.show()


if __name__ == "__main__":
    np.random.seed(666)
    data = load_data("../Data/2022-07-26_test4_with-classification.csv")
    data = format_columns(
        data,
        drop_columns=[
            "calibration_factor",
            "capture_id",
            "capture_x",
            "capture_y",
            "group_id",
            "image_height",
            "image_width",
            "uuid",
            "elapsed_time",
        ],
    )
    data["class"] = [1 if x == "Cyanobacteria" else 0 for x in data["class"]]
    data_training = data.sample(frac=0.5)
    data_training_response = data_training["class"]
    data_test = data.sample(frac=0.5)
    data_test_response = data_test["class"]
    for data in [data_test, data_training]:
        data.drop("class", axis=1, inplace=True)
    scaler = StandardScaler()
    data_training = pd.DataFrame(
        scaler.fit_transform(data_training), columns=data_training.columns
    )
    # Test data must be scaled using training data fit.
    data_test = pd.DataFrame(
        scaler.transform(data_test), columns=data_training.columns
    )

    n_iter = 10
    iters = []
    for i in range(n_iter):
        iters.append(
            resample(data_training, data_training_response, random_state=i)
        )
        print(iters[i][0].isna().sum().sum())
    iters = list(zip(*iters))
    data_training = pd.concat(iters[0])
    data_training_response = pd.concat(iters[1])

    grid = [
        [math.exp(x), math.exp(y)]
        for x, y in (
            it.product(
                np.linspace(math.log(0.05), math.log(1.0), 20),
                np.linspace(math.log(0.05), math.log(1.0), 20),
            )
        )
    ]
    grid_results = grid_search(
        grid,
        X=data_training,
        X_test=data_test,
        y=data_training_response,
        y_test=data_test_response,
    )

    grid_results_melt = grid_results.melt(
        id_vars=["C", "l1_ratio", "no_features_kept"],
        value_vars=["percent_correct", "likelihood"],
    )

    # sns.lineplot(
    #     grid_results_melt, x="no_features_kept", y="value", hue="variable"
    # )
    # plt.show()

    # scatterplot(test_poly, data_test_response, grid_results.iloc[0, -1])
    keep_pruned = SelectFromModel(
        estimator=grid_results["model"].iloc[-1]
    ).get_support()

    training_poly_model = PolynomialFeatures().fit(
        data_training.loc[:, keep_pruned]
    )
    training_poly = pd.DataFrame(
        training_poly_model.transform(data_training.loc[:, keep_pruned]),
        columns=training_poly_model.get_feature_names_out(),
    )
    test_poly_model = PolynomialFeatures().fit(data_test.loc[:, keep_pruned])
    test_poly = pd.DataFrame(
        test_poly_model.transform(data_test.iloc[:, keep_pruned]),
        columns=test_poly_model.get_feature_names_out(),
    )

    grid_results_poly = grid_search(
        grid,
        X=training_poly,
        X_test=test_poly,
        y=data_training_response,
        y_test=data_test_response,
    )

    grid_results_poly_melt = grid_results_poly.melt(
        id_vars=["C", "l1_ratio", "no_features_kept"],
        value_vars=["percent_correct", "likelihood"],
    )

    # sns.lineplot(
    #     grid_results_poly_melt.loc[
    #         grid_results_poly_melt["variable"] == "likelihood", :
    #     ],
    #     x="no_features_kept",
    #     y="value",
    #     hue="variable",
    # )
    # plt.show()

    data_test_2 = pd.read_csv("../Data/2022-11-15_classified.csv")
    data_test_2 = format_columns(
        data_test_2,
        drop_columns=[
            "calibration_factor",
            "capture_id",
            "capture_x",
            "capture_y",
            "group_id",
            "image_height",
            "image_width",
            "uuid",
            "elapsed_time",
        ],
    )
    data_test_2["class"] = [
        1 if x == "Cyanobacteria" else 0 for x in data_test_2["class"]
    ]
    data_test_2_response = data_test_2["class"]
    data_test_2.drop("class", axis=1, inplace=True)
    data_test_2 = pd.DataFrame(
        scaler.transform(data_test_2), columns=data_test_2.columns
    )
    test_2_poly_model = PolynomialFeatures().fit(
        data_test_2.loc[:, keep_pruned]
    )
    test_2_poly = pd.DataFrame(
        test_2_poly_model.transform(data_test_2.loc[:, keep_pruned]),
        columns=test_2_poly_model.get_feature_names_out(),
    )
    breakpoint()
    model = grid_results_poly["model"][0]
    test_predict = model.predict(test_2_poly)
    print(test_predict)
    test_correct = data_test_2_response == test_predict
    print(f"percent correct = {test_correct.sum() / data_test_2.shape[0]}")
    breakpoint()
