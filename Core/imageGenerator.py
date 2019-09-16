
import os
import time
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

def loadSubdirectoryImages(directory):
    images = []
    for path, dir, files in os.walk(directory):
        for filename in files:
            # print(path, dir, filename)
            print(filename)
            image = cv2.imread(os.path.join(path, filename))
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
            cv2.imwrite(os.path.join(ComposedImage_DIRECTORY,
                                     "object" + str(objectIndex) +
                                     "background" + str(backgroundIndex)
                                     + ".jpg"), composedImage)

def objectAugmentation(objects, doAugmentation):
    if doAugmentation == "True":
        print("Object Augmentation Started")
        augmentationFlip(objects, Object_Augmented_DIRECTORY, "object", doAugmentation=doObjectFlip)
        augmentationScale(objects, Object_Augmented_DIRECTORY, doAugmenation=doObjectScale)
        augmentationBright(objects, Object_Augmented_DIRECTORY, "object", doAugmentation=doObjectBrightness)
        print("Object Augmentation Finished")
    else:
        print("Do Not Object Augmentation")

def backgroundAugmentation(backgrounds, doAugmenation):
    if doAugmenation == "True":
        print("Background Augmentation Started")
        augmentationFlip(backgrounds, Background_Augmented_DIRECTORY, "background", doAugmentation=doBackgroundFlip)
        augmentationBright(backgrounds, Background_Augmented_DIRECTORY, "background", doAugmentation=doBackgroundBrightness)
        print("Background Augmentation Finished")
    else:
        print("Do Not Background Augmentation")

def composedAugmentation(composed, doAugmnetation):
    if doAugmnetation == "True":
        print("Composed Augmentation Started")

        augmentationWeather(composed, ComposedImage_Augmented_DIRECTORY, doAugmentation=doComposedWeather)
        augmentationBright(composed, ComposedImage_Augmented_DIRECTORY, "composed", doAugmentation=doComposedBrightness)
        augmentationNoise(composed, ComposedImage_Augmented_DIRECTORY, doAugmentation=doComposedNoise)
        print("Composed Augmentation Finish")

    else:
        print("Do Not Composed Image Augmentation")


def augmentationFlip(images, saveDirectory, imageType, doAugmentation):
    if doAugmentation == "True":
        if imageType == "object":
            metafile = open(os.path.join(Object_DIRECTORY, imageType + "_flip.txt"), 'w+')
        elif imageType == "background":
            metafile = open(os.path.join(Background_DIRECTORY, imageType + "_flip.txt"), 'w+')

        for imageIndex, image in enumerate(images):
            imageClone = image.copy()
            imageFlipped = mirroring(imageClone)
            print("Image Flipped: ", imageType, imageIndex)
            cv2.imwrite(os.path.join(saveDirectory, "flip", imageType + str(imageIndex) + ".jpg"), imageFlipped)
            metafile.write("flip/" + imageType + str(imageIndex) + ".jpg" + "\t\t\t" + "has been fliped\n")
        print("Flip Finished")
    else:
        print("No Not Flip")

    metafile.close()

def augmentationScale(images, saveDirectory, doAugmenation):
    if doAugmenation == "True":
        metafile = open(os.path.join(Object_DIRECTORY, "object_flip.txt"), 'w+')
        for imageIndex, image in enumerate(images):
            for count, scaleRate in enumerate(ScaleSet):
                imageScaled = scale(image, scale_rate=float(scaleRate))
                print("Image Scaling: Object ", imageIndex, "to Scale Rate ", scaleRate)
                cv2.imwrite(os.path.join(saveDirectory,
                                        "scale", "object" + str(imageIndex) +
                                        "Scale"+ str(count) + ".jpg"), imageScaled)
                metafile.write("scale/object" + str(imageIndex) + "Scale" + str(count) + ".jpg" + "\t\t\t" +
                                        "%s has been scaled\n" % scaleRate)
        print("Scale Finished")
    else:
        print("Do Not Scale")

    metafile.close()

def augmentationNoise(images, saveDirectory, doAugmentation):
    if doAugmentation == "True":
        metafile = open(os.path.join(ComposedImage_DIRECTORY, "composed_noise.txt"), 'w+')
        for imageIndex, image in enumerate(images):
            for count, noise in enumerate(NoiseSet):
                print("Noise: ", imageIndex, ", ", noise)
                imageNoise = noisy(noise, image)
                cv2.imwrite(os.path.join(saveDirectory,
                                     "noise", "composed" + str(imageIndex) +
                                     noise+ str(count) + ".jpg"), imageNoise)
                metafile.write("noise/composed" + str(imageIndex) + noise + str(count) + ".jpg" + "\t\t\t" +
                               "%s has been noised\n" % noise)
        print("Noise Finished")
    else:
        print("Do Not Noise")

    metafile.close()

def augmentationWeather(images, saveDirectory, doAugmentation):
    if doAugmentation == "True":
        metafile = open(os.path.join(ComposedImage_DIRECTORY, "composed_weather.txt"), 'w+')
        for imageIndex, image in enumerate(images):
            for count, weather in enumerate(WeatherSet):
                if weather == "rain":
                    for rainEffect in range(1, 8):
                        imageRain = rainy(image, rainEffect, 0.6)
                        print("Weather - Rain: Composed ", imageIndex, ", RainEffect ", rainEffect)
                        cv2.imwrite(os.path.join(saveDirectory,
                                             "weather", "composed" + str(imageIndex) +
                                             "RainEffect" + str(rainEffect) + ".jpg"), imageRain)
                        metafile.write(
                            "brightness/" + "weather/composed" + str(imageIndex) + "Rain" + str(count) + "effect" +
                            str(rainEffect) + ".jpg" + "\t\t\t" + "%s has been rained\n" % rainEffect)
                elif weather == "fog":
                    for fogEffect in range(1, 18):
                        imageFog = fog(image, fogEffect, 0.6)
                        print("Weather - Fog: Composed ", imageIndex, ", FogEffect ", fogEffect)
                        cv2.imwrite(os.path.join(saveDirectory,
                                             "weather", "composed" + str(imageIndex) +
                                             "FogEffect" + str(fogEffect) + ".jpg"), imageFog)
                        metafile.write(
                            "brightness/" + "weather/composed" + str(imageIndex) + "Fog" + str(count) + "effect"
                            + str(fogEffect) + ".jpg" + "\t\t\t" + "%s has been fogged\n" % fogEffect)
                elif weather == "snow":
                    for snowEffect in range(1, 6):
                        imageSnow = snow(image, snowEffect, 0.6)
                        print("Weather - snow: Composed ", imageIndex, ", snowEffect ", snowEffect)
                        cv2.imwrite(os.path.join(saveDirectory,
                                             "weather", "composed" + str(imageIndex) +
                                             "SnowEffect" +str(snowEffect) + ".jpg"), imageSnow)
                        metafile.write(
                            "brightness/" + "weather/composed" + str(imageIndex) + "Snow" + str(count) + "effect"
                            + str(snowEffect) + ".jpg" + "\t\t\t" + "%s has been snowed\n" % snowEffect)
        print("Weather Finished")
    else:
        print("Do Not Weather")

    metafile.close()

def augmentationBright(images, saveDirectory, imageType, doAugmentation):
    if doAugmentation == "True":
        object_discrimination = 0
        if imageType == "object":
            metafile = open(os.path.join(Object_DIRECTORY, imageType + "_brightness.txt"), 'w+')
        elif imageType == "background":
            metafile = open(os.path.join(Background_DIRECTORY, imageType + "_brightness.txt"), 'w+')
            object_discrimination = 35
        elif imageType == "composed":
            metafile = open(os.path.join(ComposedImage_DIRECTORY, imageType + "_brightness.txt"), 'w+')

        for imageIndex, image in enumerate(images):
            for count, bright in enumerate(BrightnessSet):
                print("Image Brightness: ", imageType, imageIndex, ", BrightnessRate", bright)
                imageBright = brightness_control(image, int(bright))
                cv2.imwrite(os.path.join(saveDirectory,
                                     "brightness", imageType + str(imageIndex) +
                                     "Bright" + str(count) + ".jpg"), imageBright)
                metafile.write("brightness/" + imageType + str(imageIndex) + "Bright" + str(count) + ".jpg" + "\t\t\t" +
                               "%s has been scaled\n" % bright)
        print("Brightness Finished")
    else:
        print("Do Not Brightness")

    metafile.close()

# def imageComposite(objects, backgrounds):
#     composedImages = []
#     for background in backgrounds:
#         backgroundModified = backgroundAugmentation(backgrounds)
#
#     for object in objects:
#         objectModified = objectAugmentation(object)
#
#     for eachObject in objectModified:
#         for eachBackground in backgroundModified:
#             height, width, channels = background.shape
#             composedImage = attachImage(eachBackground, eachObject, random_location(width, height))
#             cv2.imwrite(ComposedImage_DIR, composedImage)
#             composedImages.insert(composedImage)
#
#     composedAugmentation(composedImages)

if __name__ == '__main__':

    start = time.time()
    print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    FlipSet, ScaleSet, NoiseSet, WeatherSet, BrightnessSet = setParameters()

    print("Flip: ", FlipSet)
    print("Scale: ", ScaleSet)
    print("Noise: ", NoiseSet)
    print("Weather: ", WeatherSet)
    print("Brightness: ", BrightnessSet)

    doObjectAugmentation, doObjectBrightness, doObjectFlip, doObjectScale, \
    doBackgroundAugmentation, doBackgroundBrightness, doBackgroundFlip, \
    doComposedImageAugmentation, doComposedNoise, doComposedWeather, doComposedBrightness = setConfiguration()

    print(doObjectAugmentation)
    print(doObjectFlip)
    print(doObjectScale)
    print(doObjectBrightness)
    print(doBackgroundAugmentation)
    print(doBackgroundFlip)
    print(doBackgroundBrightness)
    print(doComposedImageAugmentation)
    print(doComposedNoise)
    print(doComposedBrightness)
    print(doComposedWeather)

    print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    images = loadSubdirectoryImages(Background_Crawling_DIRECTORY)

    TestImages = loadImages(TEST_IMAGE_DIRECTORY)

    ObjectImageGenerate(TestImages, model)

    print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    objectImages = loadImages(Object_DIRECTORY)

    objectAugmentation(objectImages, doAugmentation=doObjectAugmentation)

    print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    backgroundImages = loadImages(Background_DIRECTORY)

    backgroundAugmentation(backgroundImages, doAugmenation=doBackgroundAugmentation)

    print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    objectImages.extend(loadSubdirectoryImages(Object_Augmented_DIRECTORY))

    backgroundImages.extend(loadSubdirectoryImages(Background_Augmented_DIRECTORY))

    imageComposite(objectImages, backgroundImages)

    print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    objectImages.clear()
    backgroundImages.clear()

    composedImages = loadImages(ComposedImage_DIRECTORY)

    composedAugmentation(composedImages, doAugmnetation=doComposedImageAugmentation)

    print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    composedImages.clear()
