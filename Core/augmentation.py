import numpy as np
import cv2


def noisy(noise_type, image):
   if noise_type == "gauss":
      row,col,channel= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,channel))
      gauss = gauss.reshape(row,col,channel)
      noisy = image + gauss
      return noisy
   elif noise_type == "s&p":
      row,col,channel = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[tuple(coords)] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[tuple(coords)] = 0
      return out
   elif noise_type == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_type =="speckle":
      row,col,channel = image.shape
      gauss = np.random.randn(row,col,channel)
      gauss = gauss.reshape(row,col,channel)
      noisy = image + image * gauss
      return noisy


def mirroring(image):
    imageMirrored = cv2.flip(image, 1)

    return imageMirrored