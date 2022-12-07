#!/usr/bin/env python3
# Copyright 2022 Neko Juers

import numpy as np
import os
from dill import dump, load
import re
import subprocess as sp

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import multiprocess as mp
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os

# Change working path to directory file is in.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Set up find command
findCMD = "find .. -name '*_Data.csv'"
out = sp.Popen(
    findCMD, shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE
)
# Get standard out and error
(stdout, stderr) = out.communicate()
# Save found files to list
filelist = stdout.decode().split()
pattern = re.compile("_.+")
all_data = []
#
for i, file in enumerate(filelist):
    # Get date and magnification.
    path_split = file.split(os.sep)
    # Extract important parts of path.
    index_info = [path_split[3], path_split[4]]
    # Further extract important parts of important parts.
    indices = [pattern.sub("", item) for item in index_info]
    df = pd.read_csv(file)
    df.columns = (
        df.columns.str.replace(" ", "_")
        .str.replace("/", "-")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    df["magnification"] = indices[0]
    df["date"] = indices[1]
    df["run_id"] = i
    df = df.set_index(["date", "magnification"], drop=False)
    all_data.append(df)

# Make one big 'ol data frame.
data = pd.concat(all_data)
unique_dates = set(data["date"])
# Map dates to colors.
colors_date = {
    date: i for date, i in zip(unique_dates, range(len(list(unique_dates))))
}
data["color"] = [colors_date[date] for date in data["date"]]
data.to_csv("../Data/bigOlDF.csv")

# Scatter of raw data.
fig, ax = plt.subplots()
ax.scatter(data["ch1_peak"], data["ch2_peak"], c=data["color"], alpha=0.1)
ax.axline((0, 0), slope=1)
fig.show()

data = data.loc[(data["ch2_peak"] != 0) & (data["ch1_peak"] != 0), :]

# Histogram of ch2/ch1
fig2, ax2 = plt.subplots()
ax2.hist(data["ch2-ch1_ratio"], bins=1000)
plt.xlim(-0.5, 2)
fig2.show()

# Colorize mixture model.
gm = GaussianMixture(n_components=2).fit(
    data["ch2-ch1_ratio"].values.reshape(-1, 1)
)
data["pred"] = gm.predict(data["ch2-ch1_ratio"].values.reshape(-1, 1))
fig3, ax3 = plt.subplots()
ax3.scatter(data["ch1_peak"], data["ch2_peak"], c=data["pred"], alpha=0.1)
ax3.axline((0, 0), slope=1)
fig3.show()

# Filter out cyanobacteria and detritus.
fig4, ax4 = plt.subplots()
data_diatom_filtered = data[(data["pred"] == 1) & (data["ch2-ch1_ratio"] < 0)]
ax4.scatter(
    data_diatom_filtered["ch1_peak"],
    data_diatom_filtered["ch2_peak"],
    alpha=0.1,
)
ax4.axline((0, 0), slope=1)
fig4.show()

# Summary of diatom ratios.
count_total = data.groupby(data["date"]).size()
count_diatom = data_diatom_filtered.groupby(
    data_diatom_filtered["date"]
).size()
algae_data = pd.DataFrame(
    zip(count_total, count_diatom), columns=["count_total", "count_diatom"]
)
algae_data["proportion_diatoms"] = count_diatom / count_total
algae_data.to_csv("../Data/diatomSummary.csv")

# Elbow plot
# This really needs multiprocessing.
features_in = ["ch2-ch1_ratio", "aspect_ratio", "diameter_fd"]
data_numeric = data.select_dtypes(np.number)
likelihood = {"likelihood": [], "aic": [], "bic": []}
scaler = StandardScaler()

# Fit mixture model with range of numbers of components and collect likelihood measures.
string = (
    str(features_in)
    .replace(r", ", r"-")
    .replace(r"[", r"_")
    .replace(r"]", r"_")
    .replace(r"'", "")
)
range_gmm = range(1, 31)

try:
    with open("../Data/gms" + string + ".dill", "rb") as pickle_file:
        gms = load(pickle_file)
except FileNotFoundError:
    def fit_gmm(components):
        print(f"Fitting model with {components} components.")
        a = GaussianMixture(
            n_components=components, n_init=10, max_iter=1000, warm_start=True
        ).fit(
            data_numeric[features_in]  # .values.reshape(-1, 2)
        )
        return a
    with mp.Pool(mp.cpu_count()) as pool:
        gms = pool.map(fit_gmm, range_gmm)
    with open("../Data/gms" + string + ".dill", "wb") as file:
        dump(gms, file)

likelihood = {"likelihood": [], "aic": [], "bic": []}
for model in gms:
    likelihood["likelihood"].append(model.score(data[features_in]))
    likelihood["aic"].append(model.aic(data[features_in]))
    likelihood["bic"].append(model.bic(data[features_in]))

# Create elbow plot.
likelihood_df = pd.DataFrame(
    scaler.fit_transform(
        # Need to create dataframe within dataframe
        pd.DataFrame.from_dict(likelihood, orient="columns")
    ),
    columns=likelihood.keys(),
)
fig5, ax5 = plt.subplots()
ax5.plot(
    likelihood_df,
    linestyle="-",
    label=likelihood_df.columns,
)
ax5.legend()
ax5.set_xticks(range_gmm)
fig5.show()

data["pred"] = gms[7].predict(data[features_in])
fig6, ax6 = plt.subplots()
colors_dict = {
    i: colors.to_hex(color, keep_alpha=False)
    for i, color in enumerate(
        cm.rainbow(np.linspace(0, 1, num=gms[7].n_components))
    )
}
data["color"] = data.apply(lambda x: colors_dict[x["pred"]], axis=1)
ax6.scatter(
    data["ch2-ch1_ratio"], data["aspect_ratio"], c=data["color"], alpha=0.1
)

# data cleaning
scaler2 = StandardScaler()
pca = PCA()
data_numeric = data.select_dtypes(np.number).drop(
    [
        "capture_id",
        "group_id",
        "image_x",
        "image_y",
        "source_image",
        "pred",
        "capture_x",
        "capture_y",
        "image_height",
        "image_width",
    ],
    axis=1,
)
data_scaled = scaler2.fit_transform(data_numeric)
pca_data = pca.fit_transform(data_scaled)

fig7, ax = plt.subplots(1, 3, constrained_layout=True)
# pca
ax[0].scatter(pca_data[:, 0], pca_data[:, 1], c=data["color"], alpha=0.005)
ax[0].set_xlim([-10, 10])
ax[0].set_ylim([-10, 10])
ax[0].set_ylabel("pc2")
ax[0].set_xlabel("pc1")
# ch2/ch1 scatter
ax[1].scatter(
    data_numeric.iloc[:, data_numeric.columns == "ch1_peak"],
    data_numeric.iloc[:, data_numeric.columns == "ch2_peak"],
    c=data["color"],
    label=data["pred"],
    alpha=0.05,
)
ax[1].set_ylabel("ch2")
ax[1].set_xlabel("ch1")
# janky legend
ax[2].scatter(
    [0 for i in colors_dict.keys()],
    colors_dict.keys(),
    c=colors_dict.values(),
)
ax[2].set_xlim(-0.5, 2)
fig7.set_size_inches(11, 6)

fig7.show()
