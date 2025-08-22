from PIL.ImageChops import difference
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def characteristicFunction(condition:bool ,whatToReturn=1):
        """ for eg 1{x=2} else 0"""
        assert type(condition) == bool or np.bool, F"the condition var should be bool, but we got {type(condition)} and it is {condition}" 
        assert type(whatToReturn) == float or int, F"the whatToReturn var should be float or int" 
        if condition:
            return whatToReturn
        else:
            return 0

def init_centroids(num_clusters:int, image:np.ndarray)->np.ndarray:
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
    assert num_clusters <= image.size // image.shape[-1], F"The number of clusters ({num_clusters}) must be less than or equal to the total number of pixels ({image.size // image.shape[-1]})."
    assert isinstance(num_clusters, int), "num_clusters must be an integer."
    assert isinstance(image, np.ndarray), "image must be a numpy array."
    assert image.ndim == 3, "image must be a 3D array (H, W, C)."
    assert num_clusters > 0, "num_clusters must be greater than 0."

    H, W, C = image.shape

    assert C == 3, F" we except the image to have 3 channels"

    print(F"the image has height:{H}, width:{W} and Channels(R,G,B):{C} ")
    num_pixels = H * W
    assert num_clusters <= num_pixels, f"num_clusters:{num_clusters} cannot be greater than the number of pixels:{num_pixels}."

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(num_pixels, C)

    random_indices = np.random.choice(num_pixels, size=num_clusters, replace=False)

    # Use the random indices to get the corresponding pixel values
    centroids_init = pixels[random_indices]

    return centroids_init.astype('int')
    # *** END YOUR CODE ***



def update_centroids(centroids:np.ndarray, image:np.ndarray, max_iter=30, print_every=10):
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
    image_height, image_width, image_channels = image.shape
    reshaped_imgae =image.reshape(image_height * image_width, image_channels)
    # C^(i)  for every pixel
    cluster_index = np.zeros((image_height * image_width, 1))
    for i in range(max_iter):
        at_iteration = i
        # all_close =np.allclose(new_centroids, centroids)
        # print(F"at mew_{i} are we allclose {all_close}")
        # if all_close: return centroids  
        # centroids = new_centroids
        i =0
        # calculating C^(i)
        print(F"  at iteration {at_iteration} the centroid is \n {centroids.astype(float)}")
        new_centroids = np.zeros_like(centroids).astype(float)
        for current_pixel in reshaped_imgae:
            # print(F"at {i} and image is of len {image.shape[0]} and image is of shape {image.shape} and centroid's shape is {centroids.shape} and current_pixel is of shape {current_pixel.shape}  ")
            # for every pixel we loop through the centroid to find the nearest one
            square_difference = np.square(current_pixel - centroids )
            square_difference = np.sum(square_difference, axis = 1)
            argmin_diff= np.argmin(square_difference)
            cluster_index[i] = argmin_diff
            # squared_diff = difference_squared
            # print(F" the difference is {square_difference} and it's shape is {square_difference.shape} ")
            i+=1
        # calculating mu_j for each j 
        # updating the centroids(mean)
        print(F"  at iteration {at_iteration} after the C(i) the centroid is \n {centroids.astype(float)}")
        for cluster_centroid_index in range(centroids.shape[0]):
            cluster_centroid = centroids[cluster_centroid_index]
            num = 0.0
            den = 0.0
            index = 0

            # print(F" the shape of centroids are {centroids.shape}")
            assert cluster_index.shape[0] == reshaped_imgae.shape[0]
            for pixel in reshaped_imgae:
                # cond = (cluster_index[index] == cluster_centroid_index )
                # print(F" the cluseter index is {cluster_index[index]} and cluster_centroid_index is {cluster_centroid_index}")
                num += characteristicFunction(cluster_index[index][0] == cluster_centroid_index) * pixel
                den += characteristicFunction(cluster_index[index][0] == cluster_centroid_index) 
                # if cond.all() == True:
                # print(F" the num is {num} and the den is {den}")
                index += 1
            assert den != 0, F" the deno.. can't be 0"
            mu_j = num/den
            new_centroids[cluster_centroid_index] =mu_j
            print(F"\n\n---at mew{cluster_centroid_index}  and the new_centroids is\n {new_centroids}\n-------\n\n")
            # input("Hit Enter to move forward")
        
        all_close =np.allclose(new_centroids, centroids)
        print(F"at iteration {at_iteration} are we allclose {all_close}")
        print(F"diff b/w new_centroids and centroids is \n {new_centroids - centroids}")
        if all_close: print(F"the new_centroids is \n{new_centroids}\n------- centroid is \n {centroids} \n"); return new_centroids  
        centroids = new_centroids
        print(F"------+++++")


    return new_centroids
    # Initialize `dist` vector to keep track of distance to every centroid
    # *** END YOUR CODE ***


def update_image(image:np.ndarray, centroids:np.ndarray):
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

    assert isinstance(image, np.ndarray), "image must be a numpy array."
    assert isinstance(centroids, np.ndarray), "centroids must be a numpy array."
    image_height, image_width, image_channels = image.shape
    reshaped_image =image.reshape(image_height * image_width, image_channels)
    new_img = reshaped_image.copy()
    image_index = 0
    for pixel in reshaped_image:
        # loop over each centorid to find the best fit
        assert image_index <= image_height * image_width and reshaped_image.shape[0], F"the image_index:{image_index} should not be able to exceed image size:{reshaped_image.shape[0]} "
        assert isinstance(pixel, np.ndarray), "pixel must be a numpy array."
        assert len(pixel) == image_channels, F"the lenght of pixels:{len(pixel)} should be same as the image channels(pixel should contain R,G,B in a array for eg) :{image_channels} "
        min_distance_squared = np.inf
        closest_centroid_index = 00.00
        for index_centroid in range(centroids.shape[0]):
            centroid:np.ndarray = centroids[index_centroid]
            assert isinstance(centroid, np.ndarray), "centroid must be a numpy array."
            l2_norm = np.sum((pixel - centroid)**2)
            if l2_norm < min_distance_squared:
                min_distance_squared = l2_norm
                closest_centroid_index = index_centroid

        # got the index of the closest_centroid
        print(F"for pixel the min_distance_squared is {min_distance_squared} and closest_centroid_index is {closest_centroid_index} and the pixels for new img (or centroids[closest_centroid_index]) is {centroids[closest_centroid_index]}  ")
        new_img[image_index] = centroids[closest_centroid_index]
        image_index+=1

    print(F"the new_img (before resizing ) is {new_img.shape}")
    new_image = new_img.reshape((image_height, image_width, image_channels))
    print(F"the new_image after resizing is of type {type(new_image)} and it is {new_image}")
    print(F"=== the new/updated image  is {new_image.shape}")
    return new_image
            


    # *** START YOUR CODE ***
    # *** END YOUR CODE ***



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
