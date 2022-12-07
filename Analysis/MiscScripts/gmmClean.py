#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import functools as ft
import pandas as pd
import math
import numpy as np

import matplotlib.pyplot as plt
import multiprocess as mp
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def clean_data(data):
    data_clean = data.loc[(data["ch2_peak"] != 0) & (data["ch1_peak"] != 0), :]
    return data_clean


def elbow_plot(data):
    figure()
    return sns.lineplot(
        data=data,
        x="index",
        y="value",
        hue="variable",
    )


# Fit ch2 model
def fit_gmm(i, X):
    print(f"Fitting model with {i} components.")
    return GaussianMixture(n_components=i).fit(X)


def get_likelihood(data, gmms):
    likelihood_dict = {
        i: {
            criterion: getattr(model, criterion)(data)
            for criterion in ["score", "aic", "bic"]
        }
        for i, model in enumerate(gmms)
    }
    likelihood_df = pd.DataFrame(likelihood_dict).T
    likelihood_df_scaled = pd.DataFrame(
        StandardScaler().fit_transform(likelihood_df),
        columns=likelihood_df.columns,
    ).reset_index()
    likelihood_melted = likelihood_df_scaled.melt(id_vars="index")
    likelihood_melted["index"] = likelihood_melted["index"] + 1
    return likelihood_melted


sns.set_style("darkgrid")
data_full = pd.read_csv("../Data/bigOlDF.csv")
data_clean = clean_data(data_full)

figure()
scatter = sns.scatterplot(
    data=data_clean, x="ch1_peak", y="ch2_peak", alpha=0.10
)

n_components = range(1, 11)
with mp.Pool(mp.cpu_count()) as pool:
    gmms = pool.map(
        ft.partial(
            fit_gmm, X=data_clean[["ch2_peak", "ch1_peak", "ch2-ch1_ratio"]]
        ),
        n_components,
    )

data_clean["ch2-ch1_group_predicted"] = gmms[4].predict(
    data_clean[["ch2_peak", "ch1_peak", "ch2-ch1_ratio"]]
)

# Elbow plot
likelihood_melted = get_likelihood(
    data_clean[["ch2_peak", "ch1_peak", "ch2-ch1_ratio"]], gmms
)
figure()
elbow_plot(likelihood_melted)

figure()
sns.histplot(
    data_clean, x="ch2_peak", hue="ch2-ch1_group_predicted", multiple="stack"
)

data_cleaner = data_clean[(data_clean["ch2-ch1_group_predicted"] != 2)]
figure()
sns.histplot(
    data_cleaner, x="ch2_peak", hue="ch2-ch1_group_predicted", multiple="stack"
)

figure()
sns.scatterplot(
    data_cleaner,
    x="ch1_peak",
    y="ch2_peak",
    hue="ch2-ch1_group_predicted",
    alpha=0.1,
)

n_components = range(1, 11)
with mp.Pool(mp.cpu_count()) as pool:
    gmms_cleaner = pool.map(
        ft.partial(
            fit_gmm, X=data_cleaner[["ch2_peak", "ch1_peak", "ch2-ch1_ratio"]]
        ),
        n_components,
    )

# Elbow plot
likelihood_cleaner = get_likelihood(
    data_cleaner[["ch2_peak", "ch1_peak", "ch2-ch1_ratio"]], gmms_cleaner
)
figure()
elbow_plot(likelihood_cleaner)

data_cleaner["ch2-ch1_group_predicted"] = gmms_cleaner[1].predict(
    data_cleaner[["ch2_peak", "ch2_peak", "ch2-ch1_ratio"]]
)

figure()
sns.histplot(data_cleaner, x="ch2_peak", hue="ch2-ch1_group_predicted")
figure()
sns.scatterplot(
    data_cleaner,
    x="ch1_peak",
    y="ch2_peak",
    hue="ch2-ch1_group_predicted",
    alpha=0.1,
)


#####
data_test = pd.read_csv(
    "../Data/2022-07-26_test4_with-classification.csv"
)
classification = data_test["Class"]
data_test.drop(
    [
        "Class",
        "Capture ID",
        "Capture X",
        "Capture Y",
        "Elapsed Time",
        "Image Height",
        "Image Width",
        "UUID",
    ],
    inplace=True,
    axis=1
)
data_test = data_test[data_test["Ch1 Peak"] != 0]
data_test["ch2-ch1_group_predicted"] = gmms_cleaner[1].predict(
    data_test[["Ch2 Peak",  "Ch1 Peak", "Ch2/Ch1 Ratio"]]
)
data_test["class"] = classification

figure()
sns.histplot(data_test, x="Ch2/Ch1 Ratio")

figure()
sns.histplot(data_test, x="Ch2/Ch1 Ratio")

sns.histplot(data_cleaner.sample(400), x="ch2-ch1_ratio")

figure()
sns.scatterplot(data_test, x="Ch1 Peak", y="Ch2 Peak", hue="ch2-ch1_group_predicted", style="class")
features_in = data_test[["Ch2 Peak",  "Ch1 Peak", "Ch2/Ch1 Ratio"]]
gm_cyano = fit_gmm(3, features_in)

data_test["cyano"] = (data_test["class"] == "Cyanobacteria")
data_test["cyano_id"] = (data_test["ch2-ch1_group_predicted"] == 0)
data_test["correct"] = (data_test["cyano_id"] == data_test["cyano"])
figure()
sns.scatterplot(data_test, x="Ch1 Peak", y="Ch2 Peak", hue="ch2-ch1_group_predicted", style="correct")
