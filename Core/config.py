import os

from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.samples.coco import coco
import xml.etree.ElementTree as elementree

tree = elementree.parse("config.xml")
root = tree.getroot()
# print(root.tag)

ROOT_DIR = os.path.abspath("../")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# IMAGE_DIR = os.path.join(ROOT_DIR, "Mask_RCNN/images")

TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "images", "testimages")

Object_DIR = os.path.join(ROOT_DIR, "images", "object")

Object_Augmented_DIR = os.path.join(Object_DIR, "objectAugmented")

Object_Brightness_DIR = os.path.join(Object_Augmented_DIR, "brightness")

Object_Flip_DIR = os.path.join(Object_Augmented_DIR, "flip")

Object_Scale_DIR = os.path.join(Object_Augmented_DIR, "scale")


Background_DIR = os.path.join(ROOT_DIR,"images", "background")

Background_Augmented_DIR = os.path.join(Background_DIR, "backgroundAugmented")

Background_Brightness_DIR = os.path.join(Background_Augmented_DIR, "brightness")

Background_Flip_DIR = os.path.join(Background_Augmented_DIR, "flip")


ComposedImage_DIR = os.path.join(ROOT_DIR,"images", "composedImage")

ComposedImage_Augmented_DIR = os.path.join(ComposedImage_DIR, "composedImageAugmented")




class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)

def setParameter(augmentation):
    list = [parameter.text for parameter in augmentation.iter("parameter")]
    return list

def setParameters():
    augmentations = root.find("Parameters_Augmentations").findall("Augmentation")
    for augmentation in augmentations:
        if augmentation.attrib['name'] == "flip":
            FlipSet = setParameter(augmentation)
        elif augmentation.attrib['name'] == "scale":
            ScaleSet = setParameter(augmentation)
        elif augmentation.attrib['name'] == "noise":
            NoiseSet = setParameter(augmentation)
        elif augmentation.attrib['name'] == "weather":
            WeatherSet = setParameter(augmentation)
        elif augmentation.attrib['name'] == "brightness":
            BrightnessSet = setParameter(augmentation)

    return FlipSet, ScaleSet, NoiseSet, WeatherSet, BrightnessSet

# setParameters()
# print("Flip: ", FlipSet)
# print("Scale: ", ScaleSet)
# print("Noise: ", NoiseSet)
# print("Weather: ", WeatherSet)
# print("Brightness: ", BrightnessSet)

def setConfiguration():
    targets = root.find("Run_Augmentations").findall("AugmentationTarget")
    for target in targets:
        if target.attrib["target"] == "Object":
            doObjectAugmetation = target.text
            for augmentation in target.find("Augmentations").findall("Augmentation"):
                if augmentation.attrib["name"] == "flip":
                    doObjectFlip = augmentation.text
                elif augmentation.attrib["name"] == "scale":
                    doObjectScale = augmentation.text
                elif augmentation.attrib["name"] == "brightness":
                    doObjectBrightness = augmentation.text

        elif target.attrib["target"] == "Background":
            doBackgroundAugmentation = target.text
            for augmentation in target.find("Augmentations").findall("Augmentation"):
                if augmentation.attrib["name"] == "flip":
                    doBackgroundFlip = augmentation.text
                elif augmentation.attrib["name"] == "brightness":
                    doBackgroundBrightness = augmentation.text

        elif target.attrib["target"] == "ComposedImage":
            doComposedImageAugmentation = target.text
            for augmentation in target.find("Augmentations").findall("Augmentation"):
                if augmentation.attrib["name"] == "noise":
                    doComposedNoise = augmentation.text
                elif augmentation.attrib["name"] == "brightness":
                    doComposedBrightness = augmentation.text
                elif augmentation.attrib["name"] == "weather":
                    doComposedWeather = augmentation.text

    return doObjectAugmetation, doObjectBrightness, doObjectFlip, doObjectScale, \
           doBackgroundAugmentation, doBackgroundBrightness, doBackgroundFlip, \
           doComposedImageAugmentation, doComposedNoise, doComposedWeather, doComposedBrightness


# setConfiguration()
# print(doObjectAugmetation)
# print(doObjectFlip)
# print(doObjectScale)
# print(doObjectBrightness)
# print(doBackgroundAugmentation)
# print(doBackgroundFlip)
# print(doBackgroundBrightness)
# print(doComposedImageAugmentation)
# print(doComposedNoise)
# print(doComposedBrightness)
# print(doComposedWeather)

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