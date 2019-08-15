import cv2
from Core.augmentation import noisy, mirroring

if __name__ == '__main__':


    gaussed_background = noisy("gauss", background)
    saltAndPepper_background = noisy("s&p", background)
    poissoned_background = noisy("poisson", background)
    speckled_background = noisy("speckle", background)

    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/GaussedImage.jpg"), gaussed_background)
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SaltPapperedImage.jpg"), saltAndPepper_background)
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/PossionedImage.jpg"), poissoned_background)
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SpeckledImage.jpg"), speckled_background)


    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/GaussedMirroredImage.jpg"), mirroring(gaussed_background))
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SaltPapperedMirroredImage.jpg"), mirroring(saltAndPepper_background))
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/PossionedMirroredImage.jpg"), mirroring(poissoned_background))
    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageAugmented/SpeckledMirroredImage.jpg"), mirroring(speckled_background))