import numpy as np
import matplotlib.pyplot as plt
import init_centroids
import k_means
import loader
import sys

RGB_SIZE = 3
IMG_SIZE = 128
EPOCH = 10
FIRST_ARG = 1
SEC_ARG = 2
THIRD_ARG = 3


def main():
    img = loader.load_data(sys.argv[FIRST_ARG])

    for i in range(1, 5):
        centroids = init_centroids.init_centroids(np.power(2, i))
        model = k_means.KMeans(centroids, img)
        new_img = model.algorithm(EPOCH)
        new_img = np.reshape(new_img, (int(sys.argv[SEC_ARG]), int(sys.argv[THIRD_ARG]), RGB_SIZE))
        # plot the image
        plt.imshow(new_img)
        plt.grid(False)
        plt.show()


if __name__ == "__main__":
    main()
