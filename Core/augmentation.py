import numpy as np
import glob
import cv2
import os

from Core.config import ROOT_DIR

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
    return cv2.flip(image, 1)


def crop_top_bar(images, startY, startX):
    crop_img = []
    for image in images:
        height, width, channels = image.shape
        crop_img.append(image[startY:height, startX:width])

    return crop_img


def roadview_to_background(images):
    backgrounds = crop_top_bar(images, 32, 0)
    d = 0
    for background in backgrounds:
        cv2.imwrite(os.path.join(ROOT_DIR, "images/background/bkRoad%d.jpg" % d), background)
        d += 1

    return backgrounds


if __name__ == '__main__':
    import os
    from Core.extractObject import attachImageTest

    ROOT_DIR = os.path.abspath("../")
    print(os.path.join(ROOT_DIR, "images/imageAugmented/NewImage.jpg"))

    # background = cv2.imread(os.path.join(ROOT_DIR, "images/imageGenerated/NewImage.jpg"))
    roadViews = [cv2.imread(file) for file in glob.glob(os.path.join(ROOT_DIR, "images/roadViews/*.jpg"))]

    backgrounds = roadview_to_background(roadViews)

    objects = [cv2.imread(file) for file in glob.glob(os.path.join(ROOT_DIR, "images/object/*.jpg"))]
    combinedImageSet = []
    d = 0
    for background in backgrounds:
        height, width, channels = background.shape
        for singleObject in objects:
            tmpbk = background.copy()
            attachedImage = attachImageTest(tmpbk, singleObject, int(width / 2), int(height / 2))
            combinedImageSet.append(attachedImage)
            cv2.imwrite(os.path.join(ROOT_DIR, "images/imageGenerated/GeneratedImage%d.jpg" % d), attachedImage)
            d += 1

    '''
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
    '''
