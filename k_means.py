import numpy as np

MAX_VAL = 1
NAN_IDX = -1
RGB_SIZE = 3


class KMeans:
    """
    Summary:
        K_Means algorithm.

    Members:
        centroids [np.array]: Contains the centroids pixel, 2D array.
        image [np.array]: Contains the image pixels, 3D array.

    """

    def __init__(self, centroids, image):
        self.centroids = centroids
        self.image = image

    def re_centroid(self, centroids, classified) -> None:
        """
        Summary:
            Set the new centroids as the mean of closest pixels.

        Arguments:
            centroids [np.array]: Contains the centroids pixel, 2D array.
            classified [np.array]: Contains the closest image pixels to each centroid, 2D array.

        Returns:
            None.
        """
        for i in range(len(classified)):
            centroids[i] = np.mean(classified[i], axis=0)

    def algorithm(self, epoch) -> np.array:
        """
        Summary:
            The algorithm iterate over all the image pixels,
            for each pixel it finds the closest centroid and change the pixel in the new image to his closest centroid.
            Then finds the means of each centroids closest pixels and set them as new centroids,
            repeat this opearation ten times.
            Return the new image set after the ten iterations.

        Arguments:
            epoch [int]: Number of algorithm iterations.

        Returns:
            new_image [np.array]: The compressed image, 3D array.
        """

        classified = [[np.zeros([RGB_SIZE])] for i in range(len(self.centroids))]
        new_img = np.empty_like(self.image)
        centroids_index = NAN_IDX

        for i in range(epoch):
            for p_idx, pixel in enumerate(self.image):
                shortest_distance = MAX_VAL
                for c_idx, centroid in enumerate(self.centroids):
                    curr_distance = np.linalg.norm(pixel - centroid)
                    if curr_distance < shortest_distance:
                        shortest_distance = curr_distance
                        centroids_index = c_idx
                classified[centroids_index].append(pixel)
                new_img[p_idx] = self.centroids[centroids_index]
            self.re_centroid(self.centroids, classified)

        return new_img
