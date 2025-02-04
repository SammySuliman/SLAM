import struct
import math
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dataloader import load_data

datadir = "pcd_output/velodynevlp16/data_pcl"
num_frames = len([name for name in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, name))])

points, quaternions = load_data(datadir, num_frames)
sources = []

# select the first frame
pts = points[0]

pts2 = np.array([list(tup) for tup in pts])

def plane_from_points(points):

    points = np.nan_to_num(points, nan=0.0)

    n = len(points)
    if n < 3:
        return None

    sum_vec = np.array([0.0, 0.0, 0.0])
    for p in points:
        sum_vec = sum_vec + p
    centroid = sum_vec * (1.0 / n)

    # Calculate the full 3x3 covariance matrix, excluding symmetries
    xx = xy = xz = yy = yz = zz = 0.0

    for p in points:
        r = p - centroid
        r /= np.linalg.norm(r)
        xx += r[0] * r[0]
        xy += r[0] * r[1]
        xz += r[0] * r[2]
        yy += r[1] * r[1]
        yz += r[1] * r[2]
        zz += r[2] * r[2]

    xx /= n
    xy /= n
    xz /= n
    yy /= n
    yz /= n
    zz /= n

    weighted_dir = np.array([0.0, 0.0, 0.0])

    det_x = yy * zz - yz * yz
    axis_dir = np.array([det_x, xz * yz - xy * zz, xy * yz - xz * yy])
    weight = det_x * det_x
    if weighted_dir.dot(axis_dir) < 0.0:
        weight = -weight
    weighted_dir = weighted_dir + axis_dir * weight

    det_y = xx * zz - xz * xz
    axis_dir = np.array([xz * yz - xy * zz, det_y, xy * xz - yz * xx])
    weight = det_y * det_y
    if weighted_dir.dot(axis_dir) < 0.0:
        weight = -weight
    weighted_dir = weighted_dir + axis_dir * weight

    det_z = xx * yy - xy * xy
    axis_dir = np.array([xy * yz - xz * yy, xy * xz - yz * xx, det_z])
    weight = det_z * det_z
    if np.dot(weighted_dir, axis_dir) < 0.0:
        weight = -weight
    weighted_dir = weighted_dir + axis_dir * weight

    normal = weighted_dir / np.linalg.norm(weighted_dir)
    if np.isfinite(normal[0]) and np.isfinite(normal[1]) and np.isfinite(normal[2]):
        a, b, c = normal
        centroid /= np.linalg.norm(centroid)
        d = -a * centroid[0] - b * centroid[1] - c * centroid[2]
        return a, b, c, d
    else:
        return None
    
def remove_ground_plane(points, threshold = 1):

    a, b, c, d = plane_from_points(pts)

    # Calculate distances of all points from the plane
    distances = [np.abs(a * p[0] + b * p[1] - c * p[2] + d) / np.sqrt(a**2 + b**2 + c**2) for p in pts]

    # Filter out points within the threshold distance from the plane
    indices = [index for index, value in enumerate(distances) if value > threshold]
    non_ground_points = [points[i] for i in indices]

    return non_ground_points

# NOTE : May delete layer - want to see if there is a better way to select cluster parameters rather than hardcoding
# DBSCAN parameters
# eps : Maximum distance between two points to be considered in the same neighborhood
# min_samples : Minimum number of points to form a dense region
'''
# Build NearestNeighbors model
nbrs = NearestNeighbors(n_neighbors = 2, algorithm = 'auto').fit(remaining_points)
nearest_distances = []
for pt in remaining_points:
    pt2 = np.array([pt])
    distances, indices = nbrs.kneighbors(pt2)
    nearest_point = remaining_points[indices[0][1]]
    nearest_dist = np.linalg.norm(pt2 - nearest_point)
    nearest_distances.append(nearest_dist)
'''

def remove_outliers(remaining_points, eps = 10, min_samples = 10):

    # Apply DBSCAN
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(remaining_points)

    # Extract cluster labels
    labels = clustering.labels_

    # Use Counter to count occurrences
    count = Counter(labels)

    # Print the count of each number
    '''
    for number, frequency in count.items():
        print(f"Number {number} occurs {frequency} times")
    '''
    # Identify the indices of non-outlier points
    cluster_indices = np.where(labels != -1)[0]

    # Extract the points belonging to the main cluster
    # remaining_points = np.array(remaining_points)
    # cluster_points = remaining_points[cluster_indices]
    cluster_points = {}
    for index, label in zip(cluster_indices, labels):
        cluster_points.update({remaining_points[index]: label})

    return cluster_points

def find_cluster_centers(cluster_points):
    res = {}
    for i, v in cluster_points.items():
        res[v] = [i] if v not in res.keys() else res[v] + [i]
    centers = []
    for key in res.keys():
        mean_x = np.mean([elt[0] for elt in res[key]])
        mean_y = np.mean([elt[1] for elt in res[key]])
        mean_z = np.mean([elt[2] for elt in res[key]])
        mean = np.array((mean_x, mean_y, mean_z)).reshape(1, -1)
        nbrs = NearestNeighbors(n_neighbors = len(res[key]), algorithm = 'auto').fit(res[key])
        distances, indices = nbrs.kneighbors(mean)
        cluster_center = res[key][indices[0][1]]
        centers.append(cluster_center)
    return centers

remaining_points = remove_ground_plane(pts)
cluster_points = remove_outliers(remaining_points)
center_points = find_cluster_centers(cluster_points)
# print('center points', center_points)
