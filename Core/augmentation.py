import numpy as np
import os
import cv2


def noisy(noise_typ, image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

def mirroring(image):
    imageMirrored = cv2.flip(image, 1)

    return imageMirrored

if __name__ == '__main__':
    import os

    ROOT_DIR = os.path.abspath("../")
    print(os.path.join(ROOT_DIR, "images/imageAugmented/NewImage.jpg"))

    background = cv2.imread(os.path.join(ROOT_DIR, "images/imageGenerated/NewImage.jpg"))

    gaussed_background = noisy("gauss", background)
    saltAndPepper_background = noisy("s&p", background)
    poissoned_background = noisy("poisson", background)
    speckled_background = noisy("speckle", background)
    #print(background)
    #print(noise_background)
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/GaussedImage.jpg"), gaussed_background)
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SaltPapperedImage.jpg"), saltAndPepper_background)
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/PossionedImage.jpg"), poissoned_background)
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SpeckledImage.jpg"), speckled_background)


    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/GaussedMirroredImage.jpg"), mirroring(gaussed_background))
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SaltPapperedMirroredImage.jpg"), mirroring(saltAndPepper_background))
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/PossionedMirroredImage.jpg"), mirroring(poissoned_background))
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SpeckledMirroredImage.jpg"), mirroring(speckled_background))