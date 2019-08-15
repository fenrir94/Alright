import cv2
import random
import os


def random_location(background_width, background_height, object_width, object_height):
    random_x = random.randrange(0, background_width - object_width-1)
    random_y = background_height - object_height - 1
    if(background_height/3 > object_height):
        random_y = random.randrange(int(2*background_height/3), background_height - object_height - 1)
    print(random_x, random_y)
    return random_x, random_y


def scale(object_image, scale_rate):
    if type(scale_rate) is int:
        scale_rate = float(scale_rate) / 100
    return cv2.resize(object_image, dsize=(0, 0), fx=scale_rate, fy=scale_rate)


def rainy(image, effect_number, composite_rate=0.6):
    if type(composite_rate) is int:
        composite_rate = float(composite_rate) / 100
    effect = '../images/weather/rain' + str(effect_number) + '.jpg'
    return weather_effect(image, effect, composite_rate)


def fog(image, effect_number, composite_rate=0.3):
    if type(composite_rate) is int:
        composite_rate = float(composite_rate) / 100
    effect = '../images/weather/fog' + str(effect_number) + '.jpg'
    return weather_effect(image, effect, composite_rate)


def snow(image, effect_number, composite_rate=0.6):
    if type(composite_rate) is int:
        composite_rate = float(composite_rate) / 100
    effect = '../images/weather/snow' + str(effect_number) + '.jpg'
    return weather_effect(image, effect, composite_rate)


def weather_effect(image, effect_path, composite_rate):
    effect_image = cv2.imread(effect_path, cv2.IMREAD_COLOR)
    image_height, image_width, _ = image.shape

    effect_image = cv2.resize(effect_image, dsize=(image_width, image_height))
    return cv2.addWeighted(image, composite_rate, effect_image, 1-composite_rate, 0)


def brightness_control(image, brightness_rate):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    if brightness_rate > 0:
        limit = 255 - brightness_rate
        value[value > limit] = 255
        value[value <= limit] += brightness_rate
    else:
        limit = 0 - brightness_rate
        value[value < limit] = 0
        value[value >= limit] -= -brightness_rate

    final_hsv = cv2.merge((hue, saturation, value))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def quadrant_brightness_control(image, first_quadrant, second_quadrant, third_quadrant, fourth_quadrant):
    image_height, image_width, _ = image.shape
    half_height = int(image_height/2)
    half_width = int(image_width/2)

    control_image = cv2.copyMakeBorder(image,0,0,0,0,cv2.BORDER_REPLICATE)
    control_image[:half_height, half_width:] = brightness_control(control_image[:half_height, half_width:], first_quadrant)
    control_image[:half_height, :half_width] = brightness_control(control_image[:half_height, :half_width], second_quadrant)
    control_image[half_height:, :half_width] = brightness_control(control_image[half_height:, :half_width], third_quadrant)
    control_image[half_height:, half_width:] = brightness_control(control_image[half_height:, half_width:], fourth_quadrant)
    return control_image
