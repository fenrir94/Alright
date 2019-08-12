
from Core.extractObject import *
from Core.augmentation import *
from Core.scaling import *
from Core.config import *

def loadImages(imageDirectory):
    imagefiles = os.listdir(imageDirectory)
    images = []
    for filename in imagefiles:
        images.append(cv2.imread(os.path.join(TEST_IMAGE_DIR, filename)))

    return images

def ObjectImageGenerate(images, model = config.model):
    for image, n, in images:
        results = model.detect([image], verbose=1)

        r = results[0]
        extractObject(image, r['rois'], r['masks'], r['class_ids'])

def imageComposite(objects, backgrounds):
    images = []
    for background in backgrounds:
        backgroundModified = backgroundAugmentation(backgrounds)

    for object in objects:
        objectModified = objectAugmentation(object)

    for eachObject in objectModified:
        for eachBackground in backgroundModified:
            height, width, channels = background.shape
            manipulatedImage = attachImage(eachBackground, eachObject, random_location(width, height))
            images.insert(manipulationAugmentation(manipulatedImage))

    return images

def objectAugmentation(object):
    #flip
    #scale
    #brightness
    images = []
    return images

def backgroundAugmentation(background):
    #flip
    #brightness
    images = []
    return images

def manipulationAugmentation(manipulation):
    #flip
    #weather
    #Noise
    #brightness
    images =[]
    return images

if __name__ == '__main__':
    config = config.config
    model = config.model

    TestImages = loadImages(TEST_IMAGE_DIR)

    ObjectImageGenerate(TestImages, model)

    imageComposite(loadImages(Object_DIR), loadImages(Background_DIR))

