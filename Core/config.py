import os

from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.samples.coco import coco
import xml.etree.ElementTree as elementree

tree = elementree.parse("config.xml")
root = tree.getroot()
# print(root.tag)

ROOT_DIRECTORY = os.path.abspath("../")
MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIRECTORY, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# IMAGE_DIRECTORY = "D:/evaluateImages"
# print(IMAGE_DIRECTORY)

IMAGE_DIRECTORY = root.find("Directories").find("Directory").text
print(IMAGE_DIRECTORY)

TEST_IMAGE_DIRECTORY = IMAGE_DIRECTORY + "testimages/"

# Object_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "object")
Object_DIRECTORY = IMAGE_DIRECTORY + "object/"
print(Object_DIRECTORY)
Object_Augmented_DIRECTORY = Object_DIRECTORY + "objectAugmented/"
Object_Brightness_DIRECTORY = Object_Augmented_DIRECTORY + "brightness/"
Object_Flip_DIRECTORY = Object_Augmented_DIRECTORY + "flip/"
Object_Scale_DIRECTORY = Object_Augmented_DIRECTORY + "scale/"

Background_DIRECTORY = IMAGE_DIRECTORY + "background/"
Background_Crawling_DIRECTORY = Background_DIRECTORY + "crawledImages/"
Background_Augmented_DIRECTORY = Background_DIRECTORY + "backgroundAugmented/"
Background_Brightness_DIRECTORY = Background_Augmented_DIRECTORY + "brightness/"
Background_Flip_DIRECTORY = Background_Augmented_DIRECTORY + "flip/"
Background_Scale_DIRECTORY = Background_Augmented_DIRECTORY + "scale/"

# ComposedImage_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "composedImage")
ComposedImage_DIRECTORY = IMAGE_DIRECTORY + "composedImage/"
ComposedImage_Augmented_DIRECTORY = ComposedImage_DIRECTORY +"composedImageAugmented/"
ComposedImage_Noise_DIRECTORY = ComposedImage_Augmented_DIRECTORY + "noise/"
ComposedImage_Weather_DIRECTORY = ComposedImage_Augmented_DIRECTORY + "weather/"
ComposedImage_Brightness_DIRECTORY = ComposedImage_Augmented_DIRECTORY + "brightness/"

Annotation_DIRECTORY = IMAGE_DIRECTORY + "annotation/"
Annotation_Test_DIRECTORY = Annotation_DIRECTORY + "test/"
# Annotation_Object_DIRECTORY = Annotation_DIRECTORY + "object/"
Annotation_Composed_DIRECTORY = Annotation_DIRECTORY + "composed/"


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("Folder Created - " + directory)
        else:
            print("Folder Existed - " + directory)
    except OSError:
        print("Error: Creating directory" + directory)

def generateDirectories():
    createFolder(Object_DIRECTORY)
    createFolder(Object_Augmented_DIRECTORY)
    createFolder(Object_Brightness_DIRECTORY)
    createFolder(Object_Flip_DIRECTORY)
    createFolder(Object_Scale_DIRECTORY)

    createFolder(Background_Augmented_DIRECTORY)
    createFolder(Background_Scale_DIRECTORY)
    createFolder(Background_Brightness_DIRECTORY)
    createFolder(Background_Flip_DIRECTORY)

    createFolder(ComposedImage_DIRECTORY)
    createFolder(ComposedImage_Augmented_DIRECTORY)
    createFolder(ComposedImage_Weather_DIRECTORY)
    createFolder(ComposedImage_Noise_DIRECTORY)
    createFolder(ComposedImage_Brightness_DIRECTORY)

    createFolder(Annotation_Composed_DIRECTORY)

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