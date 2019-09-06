import numpy as np
from matplotlib.pyplot import imread


def load_data(path) -> np.array:
    """
    Summary:
        Load the data.
        Read the image pixels value to a numpy array,
        normalize the value and reshape the array size.

    Arguments:
        centroids [np.array]: Contains the centroids pixel, 2D array.
        classified [np.array]: Contains the closest image pixels to each centroid, 2D array.

    Returns:
        reshaped_img [np.array]: The image pixels normalized and reshaped 1684X3.
    """
    img = imread(path)
    img = img.astype(float) / 255.
    img_size = img.shape
    reshaped_img = img.reshape(img_size[0] * img_size[1], img_size[2])
    return reshaped_img
