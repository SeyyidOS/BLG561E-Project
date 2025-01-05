from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

def apply_kmeans(data, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

def iterative_kmeans(dataframe, column_name, initial_k=3, max_iterations=10):
    results = {}  

    hierarchical_labels = np.full(dataframe.shape[0], -1, dtype=int) 

    current_indexes = dataframe.index.to_numpy() 
    iteration = 0
    cluster_counter = 0  

    while iteration < max_iterations:
        data_subset = np.stack(dataframe.loc[current_indexes, column_name].values)

        kmeans = KMeans(n_clusters=initial_k, random_state=42)
        labels = kmeans.fit_predict(data_subset)

        unique_labels, counts = np.unique(labels, return_counts=True)
        most_populated_cluster = unique_labels[np.argmax(counts)]

        for label in unique_labels:
            label_indexes = current_indexes[labels == label]
            hierarchical_labels[np.isin(dataframe.index, label_indexes)] = cluster_counter
            cluster_counter += 1

        results[iteration] = {
            "indexes": current_indexes,
            "labels": labels,
            "centroids": kmeans.cluster_centers_,
            "most_populated_cluster": most_populated_cluster,
        }

        current_indexes = current_indexes[labels == most_populated_cluster]

        if len(current_indexes) < initial_k:
            print(f"Stopping iteration {iteration} as the cluster size is too small for further splitting.")
            break

        iteration += 1

    dataframe["Final Cluster"] = hierarchical_labels
    return dataframe, results

def apply_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters

def plot_k_distance_graph(data):
    neighbors = NearestNeighbors(n_neighbors=5)  
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, 4])  

    plt.plot(distances)
    plt.title("k-Distance Graph")
    plt.xlabel("Data Points (sorted)")
    plt.ylabel("Distance to 5th Nearest Neighbor")
    plt.show()
