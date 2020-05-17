from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    centroids_init = (np.random.uniform(size = (num_clusters, image.shape[-1])) * 255).astype('int')
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # Usually expected to converge long before `max_iter` iterations
    # Initialize `dist` vector to keep track of distance to every centroid
    new_centroids = centroids.astype('float')
    dist = np.zeros((image.shape[0], image.shape[1], len(centroids)))
    # Loop over all centroids and store distances in `dist`
    def getDist(image, centroid):
        return np.sqrt(np.sum((image - centroid)**2, axis = -1))
    for num in range(max_iter):
        position_change = 0
        for i in range(len(centroids)):
            centroid = new_centroids[i]
            dist[:, :, i] = (getDist(image, centroid))
        # Find closest centroid and update `new_centroids`
        closest_centroid = dist.argmin(axis=-1)
        # Update `new_centroids`
        for i in range(len(centroids)):
            mask = (closest_centroid == i)
            closest_points = image[mask, :]
            if len(closest_points) != 0:
                new_centroid = np.mean(closest_points, axis=0)
                position_change += np.sqrt(np.sum((new_centroid - new_centroids[i])**2))
                new_centroids[i] = new_centroid
        # Convergence
        if position_change <= 5:
            print('Algorithm converged.')
            break
        if num % print_every == 0:
            print('In iteration {}, the positions of centroids changed by {}'.format(num, position_change))
    # *** END YOUR CODE ***
    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # Initialize `dist` vector to keep track of distance to every centroid
    dist = np.zeros((image.shape[0], image.shape[1], centroids.shape[0]))
    # Loop over all centroids and store distances in `dist`
    def getDist(image, centroid):
        return np.sqrt(np.sum((image - centroid)**2, axis = -1))
    for i in range(centroids.shape[0]):
        centroid = centroids[i]
        dist[:, :, i] = (getDist(image, centroid))
    # Find closest centroid and update pixel value in `image`
    closest_centroid = dist.argmin(axis=-1)
    for i in range(closest_centroid.shape[0]):
        for j in range(closest_centroid.shape[1]):
            image[i,j,:] = centroids[closest_centroid[i,j]]
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
