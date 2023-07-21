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


def compute_dispersion(data: np.ndarray, cluster_centroid: pd.DataFrame, dim: int) -> dict:

    dataset = pd.DataFrame(data)
    clusters,_ = assign_cluster(dataset, cluster_centroid)

    dataset["cluster"] = clusters
    cluster_centroid = cluster_centroid.reset_index(drop = True).reset_index()
    dataset = dataset.merge(cluster_centroid, left_on = "cluster", right_on = "index").drop(columns = ["index"])

    inertia_per_points = []
    for k in range(dataset.shape[0]):
        inertia_per_points = inertia_per_points + [euclidian_dist(np.array(dataset.iloc[k,:dim]),np.array(dataset.iloc[k,(dim + 1):])) ** 2]

    dataset["inertia"] = inertia_per_points 
    inertia_per_cluster = dataset.groupby("cluster").agg("mean")["inertia"].tolist()

    result = {"inertie intraclasse" : np.sum(inertia_per_cluster)}

    count = data.groupby("cluster").agg("count").iloc[:,:1].reset_index()
    count = count.rename(columns = {count.columns[0]: "cluster", count.columns[1]: "count"})
    weight_cluster = cluster_centroid.merge(count, left_on = "index", right_on = "cluster").drop(columns= "index")

    for k in range(weight_cluster.shape[0]):
        inertia_per_points = inertia_per_points + [
            weight_cluster["count"][k] * euclidian_dist(np.array(weight_cluster.iloc[k,:dim]),np.array(weight_cluster.iloc[k,(dim + 1):])) ** 2]
    
    result = result | {"inertie interclasse" :  np.sum(inertia_per_points) / np.sum(weight_cluster["count"])}

    return result