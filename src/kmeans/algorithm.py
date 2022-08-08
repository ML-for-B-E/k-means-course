import numpy as np
import pandas as pd
from kmeans.distances import euclidian_dist

def assign_cluster(data, centroids):
    '''
    Receives a dataframe of data and centroids and returns a list assigning each observation a centroid.
    data: a dataframe with all data that will be used.
    centroids: a dataframe with the centroids. For assignment the index will be used.
    '''

    n_observations = data.shape[0]
    centroid_assign = []
    centroid_errors = []
    k = centroids.shape[0]


    for observation in range(n_observations):

        # Calculate the errror
        errors = np.array([])
        for centroid in range(k):
            error = euclidian_dist(centroids.iloc[centroid, :2], data.iloc[observation,:2])
            errors = np.append(errors, error)

        # Calculate closest centroid & error 
        closest_centroid =  np.where(errors == np.amin(errors))[0].tolist()[0]
        centroid_error = np.amin(errors)

        # Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    return (centroid_assign,centroid_errors)


def compute_dispersion(data: np.ndarray, clusters: np.ndarray, cluster_centroid: pd.DataFrame) -> float:

    data = pd.DataFrame(data)
    data["cluster"] = clusters
    data = data.merge(cluster_centroid, on = "cluster")

    inertia_per_points = []
    for k in range(data.shape[0]):
        inertia_per_points = inertia_per_points + [euclidian_dist(np.array(data.iloc[k,:2]),np.array(data.iloc[k,3:])) ** 2]

    data["inertia"] = inertia_per_points 
    inertia_per_cluster = data.groupby("cluster").agg("sum")["inertia"].tolist()

    return np.sum(inertia_per_cluster)