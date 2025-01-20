"""
Helper plotting functions mostly inspired by:
https://stackoverflow.com/questions/59705290/edge-effects-density-2d-plot-with-kde
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0],
                                         towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1],
                                         towers[:, 1] <= bounding_box[3]))


def dataMirror(towers, bounding_box, perc=.1):
    i = in_box(towers, bounding_box)
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    xr, yr = np.ptp(towers.T[0]) * perc, np.ptp(towers.T[1]) * perc
    xmin, xmax = bounding_box[0] - xr, bounding_box[1] + xr
    ymin, ymax = bounding_box[2] - yr, bounding_box[3] + yr
    msk = (points[:, 0] > xmin) & (points[:, 0] < xmax) &\
        (points[:, 1] > ymin) & (points[:, 1] < ymax)
    points = points[msk]
    return points.T


def KDEplot(xmin, xmax, ymin, ymax, values, ax):
    kernel = stats.gaussian_kde(values, bw_method=.2)
    gd_c = complex(0, 50)
    x_grid, y_grid = np.mgrid[xmin:xmax:gd_c, ymin:ymax:gd_c]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    k_pos = kernel(positions)
    ext_range = [xmin, xmax, ymin, ymax]
    kde = np.reshape(k_pos.T, x_grid.shape)
    ax.imshow(np.rot90(kde), cmap=plt.get_cmap('Greys'), extent=ext_range)
