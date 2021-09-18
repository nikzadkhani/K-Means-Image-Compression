import streamlit as st
from sklearn.cluster import k_means
from PIL import Image
import numpy as np
from pyinstrument import Profiler
import plotly.graph_objs as pgo
import time
import code_text


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
    start_time = time.time()
    bar.progress(0)
    centroid_history = []
    current_centroids = initial_centroids
    clusters = []
    for iteration in range(n_iter):
        bar.progress(iteration / n_iter)
        centroid_history.append(current_centroids)
        clusters = find_closest_centroids(samples, current_centroids)
        current_centroids = get_centroids(samples, clusters, k)
        elapsed = time.time() - start_time
        if iteration > 0:
            total_time = (elapsed * n_iter)/iteration
            time_left.text(
                f'Time Elapsed: {elapsed:.2f} seconds  Time Remaining: {total_time - elapsed:.2f} seconds')

    time_left.text('Complete')
    bar.progress(1.0)

    return clusters, centroid_history


def compress_image():
    if image_file is None:
        k_means_container.error('Please upload an image to compress')
        return

    with Image.open(image_file) as im:
        image = im.convert('RGB')
    st.session_state.k_mean_og = image
    image_array = np.array(image)
    height, width, num_channels = image_array.shape
    image_array = image_array.reshape(-1, num_channels) / 255

    initial_centroids = choose_random_centroids(image_array, k)
    labels, centroid_history = run_k_means(
        image_array, initial_centroids, n_iter)
    final_centroids = centroid_history[-1]

    final_image = final_centroids[labels]
    final_image = final_image.reshape(height, width, num_channels) * 255
    final_image = final_image.astype(np.uint8)
    compressed_image = Image.fromarray(final_image)

    st.session_state.k_mean_compressed = compressed_image

    image_array = (image_array * 255).astype(np.uint8)
    final_centroids = (final_centroids * 255).astype(np.uint8)

    plot_k_means_result(
        image_array, final_image.reshape(-1, 3),
        final_centroids)


def plot_k_means_result(
        image_array, compressed_image_array, centroids, samples_size=100):
    num_pixels, num_channels = image_array.shape
    samples_size = min(samples_size, num_pixels)
    sub_idx = np.random.randint(0, num_pixels, size=(samples_size))
    red = image_array[sub_idx, 0]
    green = image_array[sub_idx, 1]
    blue = image_array[sub_idx, 2]

    red_center = centroids[:, 0]
    green_center = centroids[:, 1]
    blue_center = centroids[:, 2]

    sample_compressed_image = compressed_image_array[sub_idx]

    sample_colors = []
    for pixel in image_array[sub_idx]:
        sample_colors.append('rgb' + str(tuple(pixel)))

    centroid_colors = []
    for pixel in centroids:
        centroid_colors.append('rgb' + str(tuple(pixel)))

    raw_pixels = pgo.Scatter3d(
        x=red, y=green, z=blue, customdata=sample_compressed_image,
        mode='markers', marker=pgo.scatter3d.Marker(
            symbol='circle', size=2, color=sample_colors),
        name='Original Pixel',
        hovertemplate='<b>RGB</b>: %{x}, %{y}, %{z}<br>' +
                      '<b>Compressed</b>: %{customdata}<extra></extra>')
    compressed_pixels = pgo.Scatter3d(
        x=red_center, y=green_center, z=blue_center, mode='markers',
        marker=pgo.scatter3d.Marker(
            symbol='x', size=4, color=centroid_colors),
        name='Compressed Pixel',
        hovertemplate='<b>Compressed Color</b>: %{x}, %{y}, %{z}<extra></extra>')

    fig = pgo.Figure()
    fig.add_trace(raw_pixels)
    fig.add_trace(compressed_pixels)
    fig.update_layout(scene=dict(
        xaxis=dict(tickvals=[0, 50, 100, 150, 200, 255], showbackground=False),
        yaxis=dict(tickvals=[0, 50, 100, 150, 200, 255], showbackground=False),
        zaxis=dict(tickvals=[0, 50, 100, 150, 200, 255], showbackground=False),
        xaxis_title='Red',
        yaxis_title='Green',
        zaxis_title='Blue'),
        title={'text': '100 Point Sub-Sample of Image & Compressed Pixels'},
        font=dict(size=14),
        legend=dict(itemsizing='constant')
    )
    st.session_state.graph = fig


def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def naive_compression():
    if naive_image is None:
        naive_container.error('Please upload an image to compress')
        return

    with Image.open(naive_image) as im:
        img = im.convert('RGB')
    og_column.image(img, caption='Original Image')
    img_array = np.array(img)
    num_bits_per_channel = 2
    step = 256//num_bits_per_channel
    for i in range(num_bits_per_channel):
        idx = (img_array >= i * step) * (img_array < (i + 1) * step)
        img_array[idx] = (i * step) + step//2

    compressed = Image.fromarray(img_array)
    one_bit_column.image(compressed, caption='Compressed Image')


st.markdown('# Image Compression using K-Means Clustering')

original_example, naive_example, compressed_example = st.columns(3)
original_example.image('images/gothic.jpeg',
                       caption='Original Image', use_column_width=True)
naive_example.image(
    'images/naive_gothic.jpeg',
    caption='Naive Compression into 20 colors and 3 values per channel')
compressed_example.image(
    'images/compressed_gothic.jpeg',
    caption='Compressed Image to 16 colors with 20 iterations of K means clustering',
    use_column_width=True)

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

#################################
# State Initialization
#################################
for key in [
    'naive_og', 'k_mean_og', 'naive_compressed',
        'k_mean_compressed', 'graph']:
    if key not in st.session_state:
        st.session_state[key] = None


#################################
# Naive
#################################
naive_container = st.container()
naive_container.markdown('## Naive Compression')
naive_container.markdown(
    'An easy way to solve this problem is just divy up the values into\
    bins and map all the values that fall into a bin\'s range into that bin\'s\
    value. For the values in the red channel we can say everything from $0$\
    to $255$ in $16$-bits maps to just a single number $0$ in $8$-bits.')
naive_container.markdown(
    'You can see how it works by uploading an image below.\
    The image compression here is compressing each channel from $8$-bits to a\
    $1$-bit which will only allow two values per channel. This will give us\
    a total number of $8$ colors')

_, og_column, _, one_bit_column, _ = naive_container.columns([1, 4, 1, 4, 1])

with st.form('naive'):
    naive_image = st.file_uploader(
        'Upload an image to be naively compress', key='naive_upload')
    clicked = st.form_submit_button('Compress Image Naively')
    if clicked:
        naive_compression()


#################################
# k means
#################################

k_means_container = st.container()
k_means_container.markdown('## K-Means Compression')
k_means_container.markdown('A better way to compress an image is to choose \
    the colors we compress to intelligently. One way to do this is by running \
    an unsupervised learning algorithm called *K-means clustering*. K-means\
    will try to partition whatever data it is given into *K* clusters, hence\
    the name *K*-means. The way it works is we first randomly create $K$ \
    different centroids for our clusters label arbitrarily with integers from\
     $0$ to $K-1$. Then we assign the same integer cluster label to our data\
    points, in this case pixels, by assigning the cluster whose centroid is \
    closest by Eucledian distance also known as the $l_2$ norm and many other\
    names. After we assigned labels to all our data points we re-compute our \
    centroids by setting the new centroid to the average of all the data points\
    within the cluster. Both of these steps happen in one iteration of K-means.\
    We then continue to repeat assigning labels, and recomputing centroids,\
    for a certain number of iterations which hopefully would be once K-means \
    converges.')
k_means_container.markdown(
    'We know that K-means has converged when the new centroids we \
    calculate in iteration $i$ are marginally different from $i-1$. Well what\
    is marginal? Something like less than a pixel per channel would be marginal\
    since the images from both iterations would be the same.')

with k_means_container.expander('K Means Python Code'):
    st.code(code_text.CODE)

k_means_container.markdown(
    'You can try using K-means for image compression below! Increasing\
    $K$ and the number of iterations will take more time to process, but feel\
    free to adjust them with the sliders below.')

with k_means_container.form('k_means_form'):
    k = st.slider('Choose the amount of colors to compress to',
                  min_value=1, max_value=64, value=8)

    n_iter = st.slider(
        'Choose the number of iterations to run the optimization', min_value=1,
        max_value=50, value=10)

    image_file = st.file_uploader('Upload an image to compress')

    clicked = st.form_submit_button('Compress Image')

    time_left = st.empty()
    bar = st.empty()

    if clicked:
        compress_image()

if st.session_state.k_mean_og is not None and \
    st.session_state.k_mean_compressed is not None and\
    st.session_state.graph is not None:
    _, k_means_og, _, k_means_img, _ = k_means_container.columns([1, 4, 1, 4, 1])
    k_means_og.image(st.session_state.k_mean_og, caption='Original Image')
    k_means_img.image(st.session_state.k_mean_compressed, caption='Compressed Image')

    k_means_container.plotly_chart(st.session_state.graph)



k_means_container.markdown('An interesting thing to notice is that there is\
    nothing preventing different labels from converging to the same centroid,\
    so as K increases so does the likelihood that you will waste space since\
    you could have duplicate centroids taking up space in memory. Furthermore,\
    the likelihood increases if your image does not have a wide range of hues,\
    the easiest way to demonstrate is try uploading an image of a solid color,\
    or an image that has already been compressed.')
k_means_container.markdown('K means is a lot more expensive \
    than the naive method so it probably would be too wasteful in a really \
    lightweight application.')