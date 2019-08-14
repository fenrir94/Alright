
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


def imageCompositeSample(objects, backgrounds):
    images = []
    for object in objects:
        for background in backgrounds:
            backgroundClone = background.copy()
            height, width, channels = background.shape
            composedImage = attachImage(backgroundClone, object, random_location(width, height))
            cv2.imwrite(ComposedImage_DIR, composedImage)


def imageComposite(objects, backgrounds):
    composedImages = []
    for background in backgrounds:
        backgroundModified = backgroundAugmentation(backgrounds)

    for object in objects:
        objectModified = objectAugmentation(object)

    for eachObject in objectModified:
        for eachBackground in backgroundModified:
            height, width, channels = background.shape
            composedImage = attachImage(eachBackground, eachObject, random_location(width, height))
            cv2.imwrite(ComposedImage_DIR, composedImage)
            composedImages.insert(composedImage)

    composedAugmentation(composedImages)



def objectAugmentation(objects):
    augmentationFlip(objects, Object_Augmented_DIR, "object")
    augmentationScale(objects, Object_Augmented_DIR)
    augmentationBright(objects, Object_Augmented_DIR, "object")

def backgroundAugmentation(backgrounds):
    augmentationFlip(backgrounds, Background_Augmented_DIR, "background")
    augmentationBright(backgrounds, Background_Augmented_DIR, "background")

def composedAugmentation(composed):
    augmentationNoise(composed, ComposedImage_Augmented_DIR)
    augmentationWeather(composed, ComposedImage_Augmented_DIR)
    augmentationBright(composed, ComposedImage_Augmented_DIR, "composed")


def augmentationFlip(images, saveDirectory, imageType):
    for imageIndex, image in enumerate(images):
        imageClone = image.copy()
        imageFlipped = mirroring(imageClone)
        cv2.imwrite(os.path.join(saveDirectory, "flip/" + imageType + str(imageIndex)+ ".jpg"), imageFlipped)

def augmentationScale(images, saveDirectory):
    for imageIndex, image in enumerate(images):
        for count, scaleRate in enumerate(ScaleSet):
            imageScaled = scale(image, scale_rate=scaleRate)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "scale/object" + str(imageIndex) +
                                     "Scale"+ str(count) + ".jpg"), imageScaled)

def augmentationNoise(images, saveDirectory):
    for imageIndex, image in enumerate(images):
        for count, noise in enumerate(NoiseSet):
            imageNoise = noisy(noise, image)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "noise/composed" + str(imageIndex) +
                                     noise+ str(count) + ".jpg"), imageNoise)

def augmentationWeather(images, saveDirectory):
    for imageIndex, image in enumerate(images):
        for count, weather in enumerate(WeatherSet):
            if weather == "rain":
                imageRain = rainy(image, 1, 0.6)
                cv2.imwrite(os.path.join(saveDirectory,
                                     "weather/composed" + str(imageIndex) +
                                     "Rain" + str(count) + ".jpg"), imageRain)
            elif weather == "fog":
                imageFog = fog(image, 1, 0.6)
                cv2.imwrite(os.path.join(saveDirectory,
                                     "weather/composed" + str(imageIndex) +
                                     "Fog" + str(count) + ".jpg"), imageFog)
            elif weather == "snow":
                imageSnow = snow(image, 1, 0.6)
                cv2.imwrite(os.path.join(saveDirectory,
                                     "weather/composed" + str(imageIndex) +
                                     "Snow" + str(count) + ".jpg"), imageSnow)

def augmentationBright(images, saveDirectory, imageType):
    for imageIndex, image in enumerate(images):
        for count, bright in enumerate(BrightSet):
            imageBright = brightness_control(image, bright)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "brightness/" + imageType + str(imageIndex) +
                                     "Bright" + str(count) + ".jpg"), imageBright)

if __name__ == '__main__':
    config = config.config
    model = config.model

    TestImages = loadImages(TEST_IMAGE_DIR)

    ObjectImageGenerate(TestImages, model)

    imageCompositeSample(loadImages(Object_DIR), loadImages(Background_DIR))

