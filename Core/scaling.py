import cv2
import random


def random_location(image_width, image_height):
    random_x = random.randrange(0, image_width)
    random_y = random.randrange(image_height/3, image_height)
    return random_x, random_y


def scale(object_image, scale_rate):
    if scale_rate.type == int:
        scale_rate = float(scale_rate) / 100
    return cv2.resize(object_image, dsize=(0, 0), fx=scale_rate, fy=scale_rate)


def rainy(image, effect_number):
    effect = '../images/rain/rain' + str(effect_number) + '.jpg'
    effect_image = cv2.imread(effect, cv2.IMREAD_COLOR)
    image_height, image_width, _ = image.shape

    effect_image = cv2.resize(effect_image, dsize=(image_width, image_height))
    return cv2.addWeighted(image, 0.6, effect_image, 0.4, 0)


def brightness_control(image, brightness_rate):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if brightness_rate > 0:
        lim = 255 - brightness_rate
        v[v > lim] = 255
        v[v <= lim] += brightness_rate
    else:
        lim = 0 - brightness_rate
        v[v < lim] = 0
        v[v >= lim] -= -brightness_rate

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

