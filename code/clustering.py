from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import numpy as np

def kmeans_cluster(verts, texcoords):
    cluster_labels = KMeans(n_clusters=3, random_state=0, n_init="auto").fit_predict(verts)

    unique_labels = np.unique(cluster_labels)
    clusters = [[] for label in unique_labels]
    sub_texcoords = [[] for label in unique_labels]
    for i, pt in enumerate(verts):
        label = cluster_labels[i]
        clusters[label].append(pt)
        sub_texcoords[label].append(texcoords[i])
    return clusters, sub_texcoords

def dbscan_cluster(verts):
    cluster_labels = DBSCAN(eps=0.4, min_samples=300,metric='euclidean').fit_predict(verts)

    unique_labels = np.unique(cluster_labels)
    clusters = [[] for label in unique_labels if label != -1]
    for i, pt in enumerate(verts):
        label = cluster_labels[i]
        if label != -1:
            clusters[label].append(pt)
    print('num clusters:', len(clusters))
    for i, c in enumerate(clusters):
        print(f'cluster {i+1} has {len(c)} points')
    return clusters

def hdbscan_cluster(verts):
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=300,metric='euclidean').fit_predict(verts)

    unique_labels = np.unique(cluster_labels)
    clusters = [[] for label in unique_labels if label != -1]
    for i, pt in enumerate(verts):
        label = cluster_labels[i]
        if label != -1:
            clusters[label].append(pt)
    print('num clusters:', len(clusters))
    for i, c in enumerate(clusters):
        print(f'cluster {i+1} has {len(c)} points')
    return clusters


