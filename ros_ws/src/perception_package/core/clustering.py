import numpy as np


def euclidean_clustering(xyz,distance_threshold,min_points):
    num_points = len(xyz)

    labels = np.full(
        num_points,
        -1, 
        dtype=int
    )
    
    visited = np.zeros(
        num_points,
        dtype=bool
    )

    cluster_id = 0 
    for i in range(num_points):

        if visited[i]:
            continue

        visited[i] = True
        cluster = [i]
        queue = [i]
        while queue:

         current = queue.pop(0)
         distances = np.linalg.norm(
             xyz - xyz[current],
             axis=1
         )
         neighbors = np.where(
             distances <= distance_threshold
         )[0] #returns tuple we want the first array which contains the indices
         for neighbor in neighbors:
             if not visited[neighbor]:
                 visited[neighbor] = True
                 queue.append(neighbor)
                 cluster.append(neighbor)
        if len(cluster) >= min_points:
         for index in cluster:
             labels[index] = cluster_id

         cluster_id += 1
    return labels