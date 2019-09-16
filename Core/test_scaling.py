import scaling
import cv2


def test_quadrant_brightness():
    bg_image = cv2.imread('../images/background/waters.jpg', cv2.IMREAD_COLOR)
    # 밝기 조절값은 -255~255가 최대, 그 이상은 최댓값과 같게 조절
    # 입력 파라미터 object_discrimination는 0이하 입력 시 보정 없고, 1이상 입력 시 오브젝트의 최소밝기를 35로 보정
    # object_discrimination 생략 시 0
    br_image = brightness_control(bg_image, 50, 0)
    # 입력 파라미터는 이미지, 1사분면(우상), 2사분면(좌상), 3사분면(좌하), 4사분면(우하)
    bri_image = quadrant_brightness_control(bg_image, 50, 100, -50, -100)

    # 이미지 출력
    cv2.imshow("origin", bg_image)
    cv2.imshow("bri", bri_image)
    cv2.imshow("br", br_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_weather():
    bg_image = cv2.imread('../images/background/waters.jpg', cv2.IMREAD_COLOR)
    # 날씨효과 함수의 입력 파라미터 composite_rate는 0.0(날씨효과 이미지 원본 출력) ~ 1.0(입력 이미지 출력) 까지 사용 가능
    # 비 효과 함수의 입력 파라미터 effect_number는 1~7까지 사용 가능
    # composite_rate 생략 시 0.6
    rainy_image = rainy(bg_image, 1, 0.6)
    # 안개 효과 함수의 입력 파라미터 effect_number는 1~17까지 사용 가능
    # composite_rate 생략 시 0.3
    fog_image = fog(bg_image, 1, 0.3)
    # 눈 효과 함수의 입력 파라미터 effect_number는 1~5까지 사용 가능
    # composite_rate 생략 시 0.6
    snow_image = snow(bg_image, 1, 0.6)

    # 이미지 출력
    cv2.imshow("origin", bg_image)
    cv2.imshow("rainy", rainy_image)
    cv2.imshow("fog", fog_image)
    cv2.imshow("snow", snow_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_scale():
    bg_image = cv2.imread('../images/background/waters.jpg', cv2.IMREAD_COLOR)
    # 크기 조절 함수의 입력 파라미터 scale_rate는 정수 입력 시 float(scale_rate)/100 으로 변환하며 1.0이 원본 이미지 크기
    scale_image = scale(bg_image, 1.5)

    # 이미지 출력
    cv2.imshow("origin", bg_image)
    cv2.imshow("scale", scale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_object_distinction():
    obj_image = cv2.imread('../images/background/waters.jpg', cv2.IMREAD_COLOR)
    # 오브젝트 합성 보정 함수
    # 오브젝트 추출 후 사용, 입력 파라미터 object_discrimination는 생략 시 35
    bri_image = object_distinction(obj_image, 35)
    # 오브젝트 추출 전 사용, 입력 파라미터 object_discrimination는 생략 시 35
    bri_image2 = object_distinction_before(obj_image, 35)

    # 이미지 출력
    cv2.imshow("origin", obj_image)
    cv2.imshow("bri", bri_image)
    cv2.imshow("bri2", bri_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
