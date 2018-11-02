"""
Created by Sanjay at 9/14/2018

Feature:
Implementation of DBSCAN - Density
Based Spatial Clustering of Applications
with Noise
"""
import math
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def read_data(filepath):
    """Read data points from file specified by filepath
    Args: file-path (str): the path to the file to be read

    Returns: numpy.ndarray: a numpy ndarray with shape (n, d) where n is the number of data points and d is the
    dimension of the data points

    """

    X = []
    with open (filepath, 'r') as f:
        lines = f.readlines ()
        for line in lines:
            X.append ([float (e) for e in line.strip ().split (',')])
    return np.array (X)

def write_data(filepath, data):
    with open (filepath, "w+") as f:
        for e in data:
            f.write (str (e))
            f.write ('\n')

def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def search_for_neighbors(X, p, eps):
    neighbors = []

    for i in range(0, X.shape[0]):
        if round(distance(X[p], X[i]), 1) <= eps:
            neighbors.append(i)
    return neighbors

def dfs(X, labels, core_point_indices, p, neighbors, cluster_count, eps, minpts):
    labels[p] = cluster_count
    i = 0
    while i < len(neighbors):
        ni = neighbors[i]  # index of current neighbor
        if labels[ni] == -1:  # previously a Noise point, but now a border point
            labels[ni] = cluster_count
        elif labels[ni] == 0:  # yet to be explored
            labels[ni] = cluster_count
            ni_neighbors = search_for_neighbors(X, ni, eps)
            if len(ni_neighbors) >= minpts:  # it is a core point
                core_point_indices.append(ni)
                neighbors += ni_neighbors
        i += 1
    return labels, core_point_indices

def dbscan(X, eps, minpts):
    """
    dbscan function for clustering Args: X (numpy.ndarray): a numpy array of points with dimension (n, d) where n
    is the number of points and d is the dimension of the data points eps (float): eps specifies the maximum distance
    between two samples for them to be considered as in the same neighborhood minpts (int): minpts is the number of
    samples in a neighborhood for a point to be considered as a core point. This includes the point itself.

    Returns: list: The output is a list of two lists, the first list contains the cluster label of each point,
    where -1 means that point is a noise point, the second list contains the indexes of the core points from the X
    array.

    Example: Input: X = np.array([[-10.1,-20.3], [2.0, 1.5], [4.3, 4.4], [4.3, 4.6], [4.3, 4.5], [2.0, 1.6], [2.0,
    1.4]]), eps = 0.1, minpts = 3 Output: [[-1, 1, 0, 0, 0, 1, 1], [1, 4]] The meaning of the output is as follows:
    the first list from the output tells us: X[0] is a noise point, X[1],X[5],X[6] belong to cluster 1 and X[2],X[3],
    X[4] belong to cluster 0; the second list tell us X[1] and X[4] are the only two core points
    """

    labels = np.zeros(X.shape[0])  # cluster number starts from 1
    core_point_indices = []
    cluster_count = 0
    for i in range(0, X.shape[0]):
        if not(labels[i] == 0):  # label = 0 means unexplored
            continue

        neighbors = search_for_neighbors(X, i, eps)
        if len(neighbors) < minpts:
            labels[i] = -1  # Noise/Outlier
        else:
            cluster_count += 1
            core_point_indices.append(i)
            labels, core_point_indices = dfs(X, labels, core_point_indices, i, neighbors, cluster_count, eps, minpts)

    labels[labels > 0] -= 1  # change cluster numbers so that they start from 0
    labels = [int(lbl) for lbl in labels]  # convert to int

    return [labels, core_point_indices]


if __name__ == '__main__':
    datadir = r'C:\Users\Sanjay Saha\CS5228-assignments\assignment-1\DBScan\Data'
    # outdir = r'C:\Users\Sanjay Saha\CS5228-assignments\assignment-1\DBScan'
    outdir = '.'

    if len (sys.argv) != 4:
        print ("Wrong command format, please follwoing the command format below:")
        print ("python dbscan-template.py data_filepath eps minpts")
        exit (0)

    X = read_data (sys.argv[1])
    eps = float(sys.argv[2])
    minpts = int(sys.argv[3])
    # X = read_data(datadir+os.sep+'data_1.txt')
    # eps = float(0.3)
    # minpts = 15

    start_time = time.time()
    # Compute DBSCAN
    db = dbscan (X, eps, minpts)
    print("========== %s Seconds ============" % (time.time() - start_time))

    # store output labels returned by your algorithm for automatic marking
    write_data(outdir + os.sep + 'Output' + os.sep + 'labels.txt', db[0])

    # store output core sample indexes returned by your algorithm for automatic marking
    write_data(outdir + os.sep + 'Output' + os.sep + 'core_sample_indexes.txt', db[1])

    _, dimension = X.shape

    # plot the graph is the data is dimension 2
    if dimension == 2:
        core_samples_mask = np.zeros_like (np.array (db[0]), dtype=bool)
        core_samples_mask[db[1]] = True
        labels = np.array (db[0])

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len (set (labels)) - (1 if -1 in labels else 0)

        # Black removed and is used for noise instead.
        unique_labels = set (labels)
        colors = [plt.cm.Spectral (each)
                  for each in np.linspace (0, 1, len (unique_labels))]

        for k, col in zip (unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot (xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple (col),
                      markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot (xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple (col),
                      markeredgecolor='k', markersize=6)

        plt.title ('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig (outdir + os.sep + 'Output' + os.sep + 'cluster-result.png')
