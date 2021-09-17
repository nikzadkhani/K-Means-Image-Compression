import streamlit as st
from sklearn.cluster import k_means
from PIL import Image
import numpy as np
from pyinstrument import Profiler



def get_centroids(samples, clusters, k):
    """
    Find the centroid given the samples and their cluster.

    :param samples: samples.
    :param clusters: list of clusters corresponding to each sample.
    :return: an array of centroids.
    """
    centroids = np.zeros((k, 3))
    for i in range(k):
        mask = np.where(clusters == i)
        centroids[i, :] = np.mean(samples[mask], axis=0)
    return centroids


def find_closest_centroids(samples, centroids):
    """
    Find the closest centroid for all samples.

    :param samples: samples.
    :param centroids: an array of centroids.
    :return: a list of cluster_id assignment.
    """

    num_samples, _ = samples.shape
    k, _ = centroids.shape
    l2_norms = np.zeros((num_samples, k))
    for k, centroid in enumerate(centroids):
        l2_norms[:, k] = np.sqrt(np.sum((samples - centroid)**2, axis=1))

    return np.argmin(l2_norms, axis=1)


def choose_random_centroids(samples, K):
    """
    Randomly choose K centroids from samples.
    :param samples: samples.
    :param K: K as in K-means. Number of clusters.
    :return: an array of centroids.
    """
    ls = list(range(len(samples)))
    ls = np.random.permutation(ls)
    rand_centroids = np.array([samples[i, :] for i in ls[:K]])
    return rand_centroids


def run_k_means(samples, initial_centroids, n_iter):
    """
    Run K-means algorithm. The number of clusters 'K' is defined by the size of initial_centroids
    :param samples: samples.
    :param initial_centroids: a list of initial centroids.
    :param n_iter: number of iterations.
    :return: a pair of cluster assignment and history of centroids.
    """

    centroid_history = []
    current_centroids = initial_centroids
    clusters = []
    for iteration in range(n_iter):
        bar.progress(iteration/n_iter)
        centroid_history.append(current_centroids)
        print("Iteration %d, Finding centroids for all samples..." % iteration)
        clusters = find_closest_centroids(samples, current_centroids)
        print("Recompute centroids...")
        current_centroids = get_centroids(samples, clusters, k)

    return clusters, centroid_history


def compress_image():
    if image_file is None:
        return

    with Image.open(image_file) as im:
        image = im.convert('RGB')
        

    image_array = np.array(image)
    height, width, num_channels = image_array.shape
    image_array = image_array.reshape(-1, num_channels)/255

    initial_centroids = choose_random_centroids(image_array, k)
    clusters, centroid_history = run_k_means(image_array, initial_centroids, n_iter)
    final_centroids = centroid_history[-1]


    final_image = final_centroids[clusters]
    final_image = final_image.reshape(height, width, num_channels)*255
    final_image = final_image.astype(np.uint8)
    final_image = Image.fromarray(final_image)

    st.image(final_image)


k = st.slider('Choose the amount of colors to compress to',
              min_value=1, max_value=64, value=8)

n_iter = st.slider('Choose the number of iterations to run the optimization',
                   min_value=1, max_value=50, value=10)

image_file = st.file_uploader('Upload an image to compress')

st.button('Compress Image', on_click=compress_image)
bar = st.progress(0)
