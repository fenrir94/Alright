
import cv2
import numpy as np
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255*np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def apply_mask_inverse(image, mask, color):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask != 1,
            image[:, :, n] * 0,
            image[:, :, n]
        )
    return image

object_count = 0

def extractObject(image, boxes, masks, class_ids):
    global object_count

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = random_colors(N)

    color = colors[0]
    clone = image.copy()

    y1, x1, y2, x2 = boxes[0]
    mask = masks[:, :, 0]

    image_masked = apply_mask_inverse(clone, mask, color)
    roi = image_masked[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(ROOT_DIR, "images/object/Object"+str(object_count)+".jpg"), roi)

    object_count+=1


j = 0;

def extractObjects(image, boxes, masks, class_ids):
    global j

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = random_colors(N)

    for i, color in enumerate(colors):
        clone = image.copy()

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]

        image_masked = apply_mask_inverse(clone, mask, color)
        roi = image_masked[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(ROOT_DIR, "images/object/Object"+str(j)+str(i)+".jpg"), roi)

    j=j+1

    return image

def attachImageTest(background_image, object_image, x, y):
    rows, cols, channels = object_image.shape

    roi = background_image[x:rows+x, y:cols+y]

    obj2gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
    ret, mask_cv = cv2.threshold(obj2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask_cv)

    obj_fg = cv2.bitwise_and(object_image, object_image, mask=mask_cv)
    back_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    dst = cv2.add(obj_fg, back_bg)

    background_image[x:rows+x,y:cols+y] = dst

    cv2.imwrite(os.path.join(ROOT_DIR, "images/imageGenerated/NewImage.jpg"), background_image)


if __name__ == '__main__':
    import os
    import random

    ROOT_DIR = os.path.abspath("../")
    print(ROOT_DIR)

    import Mask_RCNN.samples.coco.coco as coco

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    IMAGE_DIR = os.path.join(ROOT_DIR, "Mask_RCNN/images")

    TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "images/testimages")

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )

    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = [
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    imagefiles = os.listdir(TEST_IMAGE_DIR)

    for filename in imagefiles:
        image = cv2.imread(os.path.join(TEST_IMAGE_DIR,filename))
        results = model.detect([image], verbose=1)

        r = results[0]
        extractObject(image, r['rois'], r['masks'], r['class_ids'])



    # file_names = next(os.walk(IMAGE_DIR))[2]
    # image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    # file_name = "car01.jpg"
    # image = cv2.imread(os.path.join(TEST_IMAGE_DIR, file_name))
    #
    # results = model.detect([image], verbose=1)
    #
    # r = results[0]
    # extractObject(image, r['rois'], r['masks'], r['class_ids'])

    #imgobject = cv2.imread(os.path.join(ROOT_DIR, "images/object/Object.jpg"))
    #background = cv2.imread(os.path.join(ROOT_DIR,"images/background/waters.jpg"))

    #attachImageTest(background, imgobject, 100, 200)
