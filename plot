import matplotlib.pyplot as plt


def plot(new_img):
    """
    Summary:
        Plot the compressed image.

    Arguments:
        new_img [np.array]: Contains the compressed data, 3D array.

    Returns:
        None
    """
    plt.imshow(new_img)
    plt.grid(False)
    plt.show()
