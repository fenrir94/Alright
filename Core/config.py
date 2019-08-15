import os

from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.samples.coco import coco

ROOT_DIR = os.path.abspath("../")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# IMAGE_DIR = os.path.join(ROOT_DIR, "Mask_RCNN/images")

TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "images", "testimages")

Object_DIR = os.path.join(ROOT_DIR, "images", "object")

Background_DIR = os.path.join(ROOT_DIR,"images", "background")

ComposedImage_DIR = os.path.join(ROOT_DIR,"images", "composedImage")

Object_Augmented_DIR = os.path.join(ROOT_DIR, "images", "object","objectAugmented")

Background_Augmented_DIR = os.path.join(ROOT_DIR, "images","background", "backgroundAugmented")

ComposedImage_Augmented_DIR = os.path.join(ROOT_DIR, "images", "composedImage", "composedImageAugmented")

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)

ScaleSet = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]

NoiseSet = ["gauss", "s&p", "poisson", "speckle"]

WeatherSet = ["rain", "fog", "snow"]

BrightSet = [-150, -100, -50, 0, 50, 100, 150]

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