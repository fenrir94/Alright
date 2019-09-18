import os

from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.samples.coco import coco
import xml.etree.ElementTree as elementree

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Folder Created - " + directory)
    except OSError:
        print("Error")

tree = elementree.parse("config.xml")
root = tree.getroot()
# print(root.tag)

ROOT_DIRECTORY = os.path.abspath("../")
MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIRECTORY, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# IMAGE_DIRECTORY = os.path.join("D:", "images")
# print(IMAGE_DIRECTORY)

IMAGE_DIRECTORY = root.find("Directories").find("Directory").text
print(IMAGE_DIRECTORY)

TEST_IMAGE_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "testimages")

Object_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "object")
Object_Augmented_DIRECTORY = os.path.join(Object_DIRECTORY, "objectAugmented")
Object_Brightness_DIRECTORY = os.path.join(Object_Augmented_DIRECTORY, "brightness")
Object_Flip_DIRECTORY = os.path.join(Object_Augmented_DIRECTORY, "flip")
Object_Scale_DIRECTORY = os.path.join(Object_Augmented_DIRECTORY, "scale")

Background_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "background")
Background_Crawling_DIRECTORY = os.path.join(Background_DIRECTORY, "crawledImages")
Background_Augmented_DIRECTORY = os.path.join(Background_DIRECTORY, "backgroundAugmented")
Background_Brightness_DIR = os.path.join(Background_Augmented_DIRECTORY, "brightness")
Background_Flip_DIRECTORY = os.path.join(Background_Augmented_DIRECTORY, "flip")

ComposedImage_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "composedImage")
ComposedImage_Augmented_DIRECTORY = os.path.join(ComposedImage_DIRECTORY, "composedImageAugmented")

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIRECTORY, config=config
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