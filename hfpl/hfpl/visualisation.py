# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from torch.nn import Linear
from sklearn.decomposition import PCA
from seaborn import scatterplot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Custom alpha values for 0 and 1
alpha_0 = 0
alpha_1 = 0.5  # 50% opacity

# Colormap 1: Red to Transparent
colors1 = [(1, 0, 0, alpha_0), (1, 0, 0, alpha_1)]
cmap1 = mcolors.ListedColormap(colors1)

# Colormap 2: Green to Transparent
colors2 = [(0, 1, 0, alpha_0), (0, 1, 0, alpha_1)]
cmap2 = mcolors.ListedColormap(colors2)

# Colormap 3: Blue to Transparent
colors3 = [(0, 0, 1, alpha_0), (0, 0, 1, alpha_1)]
cmap3 = mcolors.ListedColormap(colors3)

# Colormap 4: Cyan to Transparent
colors4 = [(0, 1, 1, alpha_0), (0, 1, 1, alpha_1)]
cmap4 = mcolors.ListedColormap(colors4)

# Colormap 5: Magenta to Transparent
colors5 = [(1, 0, 1, alpha_0), (1, 0, 1, alpha_1)]
cmap5 = mcolors.ListedColormap(colors5)

color_array = [
    cmap1,
    cmap2,
    cmap3,
    cmap4,
    cmap5,
]

def get_grid_of_points(x_min, x_max, y_min, y_max, n_points):
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(y_min, y_max, n_points)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def inverse_pca_and_apply(func, pca, xx, yy):
    grid = np.stack([xx.flatten(), yy.flatten()]).T
    grid = pca.inverse_transform(grid)
    preds = func(grid).T
    print(preds.shape)
    preds = preds.reshape(preds.shape[0], xx.shape[0], xx.shape[1])
    print(preds.shape)
    return preds

def plot_decision_boundary(
    infer_func, pca, x_min, x_max, y_min, y_max, n_points=100, ax=None
):
    xx, yy = get_grid_of_points(x_min, x_max, y_min, y_max, n_points)
    preds = inverse_pca_and_apply(infer_func, pca, xx, yy)
    print(preds.shape)
    preds = preds.argmax(axis=0)
    print(preds.shape)
    if ax is None:
        ax = plt.gca()
    for i in set(preds.flatten()):
        ax.contourf(xx, yy, 1*(preds == i), cmap=color_array[i])
