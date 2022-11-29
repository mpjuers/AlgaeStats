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
            max_iter=1000,
        )
        for x in params
    ]
    out = np.empty((len(params), 5), dtype=float)
    models_out = []
    for i, model in enumerate(models):
        model.fit(X, y)
        model_selected_features = SelectFromModel(estimator=model)
        # Extract boolean of kept features and select
        features_to_keep = model_selected_features.get_support()
        model_test_features_kept = X_test.loc[:, features_to_keep]
        # Cross-validate and evaluate model descriptors
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
    breakpoint()
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
    out = out.sort_values(
        "percent_correct", ascending=False, ignore_index=True
    )
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
    data = pd.read_csv("../Data/2022-11-15_classified.csv")
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
            "ch2-ch1_ratio",
            "aspect_ratio",
        ],
    )
    # process data
    data["class"] = [1 if x == "Cyanobacteria" else 0 for x in data["class"]]
    response = data["class"]
    data.drop("class", axis=1, inplace=True)
    data_training = data.sample(frac=0.5)
    data_test = data.drop(data_training.index)
    # scale data
    scaler = StandardScaler().fit(data_training)
    data_training, data_test = [
        pd.DataFrame(scaler.transform(x), columns=data.columns)
        for x in [data_training, data_test]
    ]
    training_response, test_response = [
        response[x.index] for x in (data_training, data_test)
    ]

    # generate grid search
    grid = [
        [math.exp(x), math.exp(y)]
        for x, y in (
            it.product(
                np.linspace(math.log(0.01), math.log(1.0), 100),
                np.linspace(math.log(0.01), math.log(1.0), 100),
            )
        )
    ]
    grid_results = grid_search(
        grid,
        X=data_training,
        X_test=data_test,
        y=training_response,
        y_test=test_response,
    )
    grid_results_melt = grid_results.melt(
        id_vars=["C", "l1_ratio", "no_features_kept"],
        value_vars=["percent_correct", "likelihood"],
    )

    # polynomial grid search
    keep_features = data.columns[
        SelectFromModel(estimator=grid_results.iloc[0, -1]).get_support()
    ]
    poly_fit = PolynomialFeatures().fit(data_training.loc[:, keep_features])
    training_poly = pd.DataFrame(
        poly_fit.transform(
            data_training.loc[:, keep_features],
        ),
        columns=poly_fit.get_feature_names_out(),
    )
    test_poly = pd.DataFrame(
        poly_fit.transform(
            data_test.loc[:, keep_features],
        ),
        columns=poly_fit.get_feature_names_out(),
    )

    grid_results_poly = grid_search(
        grid,
        X=training_poly,
        X_test=test_poly,
        y=training_response,
        y_test=test_response,
    )
    grid_results_poly_melt = grid_results.melt(
        id_vars=["C", "l1_ratio", "no_features_kept"],
        value_vars=["percent_correct", "likelihood"],
    )
    print(
        f"polynomial percent correct: {grid_results_poly['percent_correct'][0]}"
    )
    print(f"monomial percent correct: {grid_results['percent_correct'][0]}")

    data_new = pd.read_csv("../Data/2022-07-26_test4_with-classification.csv")
    data_new = format_columns(
        data_new,
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
            "ch2-ch1_ratio",
            "aspect_ratio",
        ],
    )
    # process data
    data_new["class"] = [
        1 if x == "Cyanobacteria" else 0 for x in data_new["class"]
    ]
    response_new = data_new["class"]
    data_new.drop("class", axis=1, inplace=True)
    data_new = pd.DataFrame(
        scaler.transform(data_new), columns=scaler.get_feature_names_out()
    )
    model_new = LogisticRegression(penalty="none").fit(
        data_new.loc[:, keep_features], response_new
    )
    predict_new = model_new.predict(data_new.loc[:, keep_features])
    correct_new = response_new == predict_new
    print(
        f"percent correct (no interactions): {correct_new.sum() / data_new.shape[0]}"
    )
