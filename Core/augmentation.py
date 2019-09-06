import numpy as np
import cv2


def noisy(noise_type, image):
    if noise_type == "gauss":
        row, col, channel = image.shape
        mean = 0
        variable = 0.1
        sigma = variable ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, channel))
        gauss = gauss.reshape(row, col, channel)
        return image + gauss
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_type == "poisson":
        values = len(np.unique(image))
        values = 2 ** np.ceil(np.log2(values))
        return np.random.poisson(image * values) / float(values)
    elif noise_type == "speckle":
        row, col, channel = image.shape
        gauss = np.random.randn(row, col, channel)
        gauss = gauss.reshape(row, col, channel)
        return image + image * gauss


def mirroring(image):
    imageMirrored = cv2.flip(image, 1)

    return imageMirrored
