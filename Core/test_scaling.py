import scaling
import cv2

def test_quadrant_brightness():
    bg_image = cv2.imread('../images/background/waters.jpg', cv2.IMREAD_COLOR)
    bri_image = quadrant_brightness_control(bg_image, 50, 100, -50, -100)

    cv2.imshow("origin", bg_image)
    cv2.imshow("bri", bri_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_weather():
    bg_image = cv2.imread('../images/background/waters.jpg', cv2.IMREAD_COLOR)
    rainy_image = rainy(bg_image, 1)
    fog_image = fog(bg_image, 1)
    snow_image = snow(bg_image, 1)

    cv2.imshow("origin", bg_image)
    cv2.imshow("rainy", rainy_image)
    cv2.imshow("fog", fog_image)
    cv2.imshow("snow", snow_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_scale():
    bg_image = cv2.imread('../images/background/waters.jpg', cv2.IMREAD_COLOR)
    bri_image = scale(bg_image, 1.5)

    cv2.imshow("origin", bg_image)
    cv2.imshow("scale", bri_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
