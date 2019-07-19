import cv2
import numpy as np
import random
import math
import extractObject


def scale(object_image, background_image):
    obj_height, obj_width, _ = object_image.shape
    bg_height, bg_width, _ = background_image.shape
    random_x = random.randrange(0, bgWidth - objWidth)
    random_y = random.randrange(math.ceil(bgHeight / 2), bgHeight - objHeight)

    distance = (randomY - bgHeight / 3) * 3 / (bgHeight - objHeight)

    scale_image = cv2.resize(obj, dsize=(0, 0), fx=distance, fy=distance)

    return attachImageTest(scale_image, background_image, random_x, random_y)


def rainy(image, effect_number):
    effect = '../images/rain/rain' + str(effect_number) + '.jpg'
    effect_image = cv2.imread(effect, cv2.IMREAD_COLOR)
    image_height, image_width, _ = image.shape

    effect_image = cv2.resize(effect_image, dsize=(bgWidth, bgHeight))
    return cv2.addWeighted(image, 0.6, effect_image, 0.4, 0)
