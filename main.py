import streamlit as st
from sklearn.cluster import k_means
from PIL import Image
import numpy as np
from pyinstrument import Profiler
import plotly.graph_objs as pgo


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
        bar.progress(iteration / n_iter)
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
    image_array = image_array.reshape(-1, num_channels) / 255

    initial_centroids = choose_random_centroids(image_array, k)
    clusters, centroid_history = run_k_means(
        image_array, initial_centroids, n_iter)
    final_centroids = centroid_history[-1]

    final_image = final_centroids[clusters]
    final_image = final_image.reshape(height, width, num_channels) * 255
    final_image = final_image.astype(np.uint8)
    final_image = Image.fromarray(final_image)

    st.image(final_image)

    # centroid, label, inertia = k_means(
    #     image_array, n_clusters=k, max_iter=n_iter)

    # new_image = centroid[label].reshape(
    #     height,
    #     width,
    #     num_channels) * 255
    # print(new_image)
    # new_image = new_image.astype(np.uint8)
    # st.image(new_image, caption='sklearn')
    sample_size = 100
    sub_idx = np.random.randint(0, height * width, (sample_size))

    image_array = (image_array * 255).astype(np.uint8)
    final_centroids = (final_centroids * 255).astype(np.uint8)

    red = image_array[sub_idx, 0]
    green = image_array[sub_idx, 1]
    blue = image_array[sub_idx, 2]

    red_center = final_centroids[:, 0]
    green_center = final_centroids[:, 1]
    blue_center = final_centroids[:, 2]

    labels = clusters[sub_idx]

    trace0 = pgo.Scatter(
        x=red,
        y=green,
        mode='markers',
        name='Original Image Color',
        marker=pgo.Marker(symbol='circle',
                          size=8,
                          color=labels, opacity=0.75))
    trace1 = pgo.Scatter(
        x=red_center,
        y=green_center,
        mode='markers',
        name='Compressed Color',
        marker=pgo.Marker(symbol='x',
                          size=12,
                          color=list(range(k))
                          ))

    data = pgo.Data([trace0, trace1])
    fig = pgo.Figure(data=data)
    st.plotly_chart(fig)


def naive_compression():
    if naive_image is None:
        return

    with Image.open(naive_image) as im:
        img = im.convert('RGB')

    img_array = np.array(img)

    for i in range(8):
        idx = (img_array >= i * 32) * (img_array < (i + 1) * 32)
        img_array[idx] = (i * 32) + 16

    compressed = Image.fromarray(img_array)
    print(np.unique(img_array))
    st.image(compressed)


st.markdown('# Image Compression using K-Means Clustering')
st.markdown('## Problem')
st.markdown(
    'Let\'s say you have a $16$-bit image. This means that every pixel,\
    has $2^{16} = 65536$ values per channel. Usually images have three channels\
    red, green and blue or RGB, so you would have $65536$ values for every \
    channel. Every pixel would then need 6 bytes of space to store the $16$-bit\
    values for each channel. This might not seem like much but it really \
    adds up as you image gets larger. For an image with a height and width\
    of $500$, it would take $1.5$ Mb of space to store the photo. That is a lot \
    of space...')
st.markdown('One way to reduce the image size is to use less values for the\
    colors. So instead of using $65536$ we could use $256$ values at $8$-bits \
    which would cut our image size in half, since we are using half the bits.\
    per pixel. How do we choose what to use for the new values?')
st.markdown('## Naive Solution')
st.markdown('An easy way to solve this problem is just divy up the values into\
    bins and map all the values that fall into a bin\'s range into that bin\'s\
    value. For the values in the red channel we can say everything from $0$\
    to $255$ in $16$-bits maps to $0$ in $8$-bits.')

naive_image = st.file_uploader('Upload an image to be naively compress')
st.button('Compress Image Naively', on_click=naive_compression)

k = st.slider('Choose the amount of colors to compress to',
              min_value=1, max_value=64, value=8)

n_iter = st.slider('Choose the number of iterations to run the optimization',
                   min_value=1, max_value=50, value=10)

image_file = st.file_uploader('Upload an image to compress')

st.button('Compress Image', on_click=compress_image)
bar = st.progress(0)
