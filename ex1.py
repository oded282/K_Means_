"""
oded ben noon.
308338219
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import math

"""
This function initial the centroids the at the first iteration.
"""


def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0., 0., 0.],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0., 0.],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


"""
This function calculate the distance between two point in third dimension.
"""


def distance_calc(x, y):
    return math.sqrt(math.pow(x[0] - y[0], 2) + math.pow(x[1] - y[1], 2) + math.pow(x[2] - y[2], 2))


"""
This function calculate the nearest centroid to specific point,
and return the index of this centroid in the array.
"""


def nearest_point(centroids, y):
    minimum = 100.0
    index = 0
    for idx, x in enumerate(centroids):
        new_min = distance_calc(x, y)
        if new_min < minimum:
            minimum = new_min
            index = idx
    return index


"""
This function create matrix and iterate over all of the pixels,
and classify each of them to the nearest centroid.
"""


def classify(pixels, centroids):
    matrix = [[] for i in range(len(centroids))]
    for raw in pixels:
        for pix in raw:
            index = nearest_point(centroids, pix)
            matrix[index].append(pix)
    return matrix


"""
This function is for testing the program with the given picture.
"""


def draw(classify_matrix, centroids):
    for index, column in enumerate(range(len(classify_matrix))):
        for point in classify_matrix[index]:
            point[0] = centroids[index][0]
            point[1] = centroids[index][1]
            point[2] = centroids[index][2]
    return


"""
This function iterate on the matrix that we built of the pixels classification.
Than we calculate the average "RGB" of the pixels classified to each of the centroids,
and update the centroids.
"""


def re_centroid(matrix, centroid):
    for index, col in enumerate(range(len(matrix))):
        red = 0
        blue = 0
        green = 0
        for i in matrix[index]:  # update the pixel
            red += i[0]
            blue += i[1]
            green += i[2]
        if len(matrix[col]) > 0:  # case the length of the matrix is greater than zero.
            centroid[index][0] = red / len(matrix[col])
            centroid[index][1] = blue / len(matrix[col])
            centroid[index][2] = green / len(matrix[col])


"""
This func function print the centroids.
"""


def print_centroids(i, centroids):
    print("iter ", i, ": ", end="")
    for centroid in centroids:
        print(np.floor(centroid * 100) / 100, ", ", end=" ")
    print()
    return


"""
The main function "read" the picture pixels and store them inside array.
Than we normalize the pixels.
Than iterate 4 times with different number of centroids (2,4,8,16),
and for each centroid we iterate 10 times. 
Each time we classify the pixels to the centroids
and than we set the centroid by the average of the classified pixels and print the new centroids.
"""


def main():
    path = 'dog.jpeg'
    for j in range(1, 5):
        print("k = ", (2 ** j), ":")
        # data preparation (loading, normalizing, reshaping)
        a = imread(path)
        a_norm = a.astype(float) / 255.
        img_size = a_norm.shape
        x = a_norm.reshape(img_size[0] * img_size[1], img_size[2])
        centroids = init_centroids(x, 2 ** j)
        matrix = [[] for i in range(len(centroids))]
        for i in range(10):
            print_centroids(i, centroids)
            matrix = classify(a_norm, centroids)
            re_centroid(matrix, centroids)
        draw(matrix, centroids)
        # plot the image
        plt.imshow(a_norm)
        plt.grid(False)
        plt.show()


if __name__ == "__main__":
    main()
