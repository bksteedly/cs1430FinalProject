from sklearn.cluster import KMeans, DBSCAN
from pointnet_segmentation import classify
import numpy as np
import open3d as o3d

def dbscan_cluster(verts):
    cluster_labels = DBSCAN(eps=0.5, min_samples=5,metric='euclidean').fit_predict(verts)
    print(cluster_labels.shape)
    print(np.unique(cluster_labels))

def cluster(verts):
    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
    kmeans.fit(verts)
    centers = kmeans.cluster_centers_

    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6 = []
    cluster7 = []
    cluster8 = []
    cluster9 = []
    cluster10 = []
    # cluster5 = []
    # colors = []
    # print("centers: " + str(centers))
    # print(verts.shape)
    for point in verts:
        # print(type(centers[0]))
        d1 = np.linalg.norm(centers[0] - point)
        d2 = np.linalg.norm(centers[1] - point)
        d3 = np.linalg.norm(centers[2] - point)
        d4 = np.linalg.norm(centers[3] - point)
        d5 = np.linalg.norm(centers[4] - point)
        d6 = np.linalg.norm(centers[5] - point)
        d7 = np.linalg.norm(centers[6] - point)
        d8 = np.linalg.norm(centers[7] - point)
        d9 = np.linalg.norm(centers[8] - point)
        d10 = np.linalg.norm(centers[9] - point)
        # d5 = np.linalg.norm(centers[4] - point)
        closest = min(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10)
        if d1 == closest:
            cluster1.append(point)
            # colors.append(np.array([255,0,0]))
        elif d2 == closest:
            cluster2.append(point)
            # colors.append(np.array([0,255,0]))
        elif d3 == closest:
            cluster3.append(point)
        elif d4 == closest:
            cluster4.append(point)
        elif d5 == closest:
            cluster5.append(point)
        elif d6 == closest:
            cluster6.append(point)
        elif d7 == closest:
            cluster7.append(point)
        elif d8 == closest:
            cluster8.append(point)
        elif d9 == closest:
            cluster9.append(point)
        else:
            cluster10.append(point)
            # colors.append(np.array([255,0,255]))
        # else:
        #     cluster5.append(point)
        #     # colors.append(np.array([0,255,255]))
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    cluster3 = np.array(cluster3)
    cluster4 = np.array(cluster4)
    cluster5 = np.array(cluster5)
    cluster6 = np.array(cluster6)
    cluster7 = np.array(cluster7)
    cluster8 = np.array(cluster8)
    cluster9 = np.array(cluster9)
    cluster10 = np.array(cluster10)
    # cluster5 = np.array(cluster5)
    print(classify(cluster1))
    print(classify(cluster2))
    print(classify(cluster3))
    print(classify(cluster4))
    print(classify(cluster5))
    print(classify(cluster6))
    print(classify(cluster7))
    print(classify(cluster8))
    print(classify(cluster9))
    print(classify(cluster10))
    # print(classify(cluster5))
    # colors = np.array(colors)
    return cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10

def main():
    # cluster1, cluster2, cluster3, cluster4 = cluster(np.load("verts_v2.npy"))
    # np.save("cluster1.npy", cluster1)
    # np.save("cluster2.npy", cluster2)

    dbscan_cluster(np.load("verts_v2.npy"))




if __name__ == '__main__':
    main()
    