
import numpy
from Core.extractObject import *
from Core.augmentation import *
from Core.scaling import *
from Core.config import *

def loadImages(imageDirectory):
    files = os.listdir(imageDirectory)
    imagefiles = [file for file in files if file.endswith(".jpg")]
    images = []
    for filename in imagefiles:
        print(filename)
        image = cv2.imread(os.path.join(imageDirectory, filename))
        images.append(image)

    return images

def ObjectImageGenerate(images, model):
    for image in images:
        results = model.detect([image], verbose=1)

        r = results[0]
        extractObject(image, r['rois'], r['masks'], r['class_ids'])


def imageComposite(objects, backgrounds):
    for objectIndex, object in enumerate(objects):
        for backgroundIndex, background in enumerate(backgrounds):
            backgroundClone = background.copy()
            object_height, object_width, _ = object.shape
            background_height, background_width, _ = background.shape
            x, y = random_location(background_width, background_height, object_width, object_height)
            composedImage = attachImage(backgroundClone, object, x, y)
            print("Image Composed: Object ", objectIndex, " in Background ", backgroundIndex)
            cv2.imwrite(os.path.join(ComposedImage_DIR,
                                     "object"+ str(objectIndex)+
                                     "background" + str(backgroundIndex)
                                     + ".jpg"), composedImage)





def objectAugmentation(objects):
    print("Object Augmentation Started")
    augmentationFlip(objects, Object_Augmented_DIR, "object")
    augmentationScale(objects, Object_Augmented_DIR)
    # augmentationBright(objects, Object_Augmented_DIR, "object")
    print("Object Augmentation Finished")

def backgroundAugmentation(backgrounds):
    print("Background Augmentation Started")
    augmentationFlip(backgrounds, Background_Augmented_DIR, "background")
    augmentationBright(backgrounds, Background_Augmented_DIR, "background")
    print("Background Augmentation Finished")

def composedAugmentation(composed):
    print("Composed Augmentation Started")

    augmentationWeather(composed, ComposedImage_Augmented_DIR)
    print("Weather Finished")
    augmentationBright(composed, ComposedImage_Augmented_DIR, "composed")
    print("Brightness Finished")

    augmentationNoise(composed, ComposedImage_Augmented_DIR)
    print("Noise Finished")
    print("Composed Augmentation Finish")


def augmentationFlip(images, saveDirectory, imageType):
    for imageIndex, image in enumerate(images):
        imageClone = image.copy()
        imageFlipped = mirroring(imageClone)
        print("Image Flipped: ", imageType, imageIndex)
        cv2.imwrite(os.path.join(saveDirectory, "flip", imageType + str(imageIndex) + ".jpg"), imageFlipped)

def augmentationScale(images, saveDirectory):
    for imageIndex, image in enumerate(images):
        for count, scaleRate in enumerate(ScaleSet):
            imageScaled = scale(image, scale_rate=scaleRate)
            print("Image Scaling: Object ", imageIndex, "to Scale Rate ", scaleRate)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "scale", "object" + str(imageIndex) +
                                     "Scale"+ str(count) + ".jpg"), imageScaled)

def augmentationNoise(images, saveDirectory):
    for imageIndex, image in enumerate(images):
        for count, noise in enumerate(NoiseSet):
            print("Noise: ", imageIndex, ", ", noise)
            imageNoise = noisy(noise, image)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "noise", "composed" + str(imageIndex) +
                                     noise+ str(count) + ".jpg"), imageNoise)

def augmentationWeather(images, saveDirectory):
    for imageIndex, image in enumerate(images):
        for weather in enumerate(WeatherSet):
            if weather == "rain":
                for rainEffect in range(1, 8):
                    imageRain = rainy(image, rainEffect, 0.6)
                    print("Weather - Rain: Composed ", imageIndex, ", RainEffect ", rainEffect)
                    cv2.imwrite(os.path.join(saveDirectory,
                                             "weather", "composed" + str(imageIndex) +
                                             "RainEffect" + str(rainEffect) + ".jpg"), imageRain)
            elif weather == "fog":
                for fogEffect in range(1, 18):
                    imageFog = fog(image, fogEffect, 0.6)
                    print("Weather - Fog: Composed ", imageIndex, ", FogEffect ", fogEffect)
                    cv2.imwrite(os.path.join(saveDirectory,
                                             "weather", "composed" + str(imageIndex) +
                                             "FogEffect" + str(fogEffect) + ".jpg"), imageFog)
            elif weather == "snow":
                for snowEffect in range(1, 6):
                    imageSnow = snow(image, snowEffect, 0.6)
                    print("Weather - snow: Composed ", imageIndex, ", snowEffect ", snowEffect)
                    cv2.imwrite(os.path.join(saveDirectory,
                                             "weather", "composed" + str(imageIndex) +
                                             "SnowEffect" +str(snowEffect) + ".jpg"), imageSnow)

def augmentationBright(images, saveDirectory, imageType):
    for imageIndex, image in enumerate(images):
        for count, bright in enumerate(BrightSet):
            print("Image Brightness: ", imageType, imageIndex, ", BrightnessRate", bright)
            imageBright = brightness_control(image, bright)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "brightness", imageType + str(imageIndex) +
                                     "Bright" + str(count) + ".jpg"), imageBright)

if __name__ == '__main__':

    TestImages = loadImages(TEST_IMAGE_DIR)

    ObjectImageGenerate(TestImages, model)

    objectImages = loadImages(Object_DIR)

    objectAugmentation(objectImages)

    backgroundImages = loadImages(Background_DIR)

    backgroundAugmentation(backgroundImages)

    objectImages.extend(loadImages(Object_Flip_DIR))
    objectImages.extend(loadImages(Object_Scale_DIR))
    objectImages.extend(loadImages(Object_Brightness_DIR))

    backgroundImages.extend(loadImages(Background_Flip_DIR))
    backgroundImages.extend(loadImages(Background_Brightness_DIR))

    imageComposite(objectImages, backgroundImages)

    composedImages = loadImages(ComposedImage_DIR)

    composedAugmentation(composedImages)



