CODE = '''
import numpy as np

def get_centroids(samples, clusters, k):
    centroids = np.zeros((k, 3))
    for i in range(k):
        mask = np.where(clusters == i)
        centroids[i, :] = np.mean(samples[mask], axis=0)
    return centroids


def find_closest_centroids(samples, centroids):
    num_samples, _ = samples.shape
    k, _ = centroids.shape
    l2_norms = np.zeros((num_samples, k))
    for k, centroid in enumerate(centroids):
        l2_norms[:, k] = np.sqrt(np.sum((samples - centroid)**2, axis=1))

    return np.argmin(l2_norms, axis=1)

def choose_random_centroids(samples, K):
    ls = list(range(len(samples)))
    ls = np.random.permutation(ls)
    rand_centroids = np.array([samples[i, :] for i in ls[:K]])
    return rand_centroids

def run_k_means(samples, initial_centroids, n_iter):
    centroid_history = []
    current_centroids = initial_centroids
    clusters = []
    for iteration in range(n_iter):
        centroid_history.append(current_centroids)
        clusters = find_closest_centroids(samples, current_centroids)
        current_centroids = get_centroids(samples, clusters, k)
    return clusters, centroid_history
'''
