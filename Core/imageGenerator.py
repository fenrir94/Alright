
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


def imageCompositeSample(objects, backgrounds):
    for objectIndex, object in enumerate(objects):
        for backgroundIndex, background in enumerate(backgrounds):
            print(objectIndex, backgroundIndex)
            backgroundClone = background.copy()
            object_height, object_width, _ = object.shape
            background_height, background_width, _ = background.shape
            x, y = random_location(background_width, background_height, object_width, object_height)
            composedImage = attachImage(backgroundClone, object, x, y)
            cv2.imwrite(os.path.join(ComposedImage_DIR,
                                     "object"+ str(objectIndex)+
                                     "background" + str(backgroundIndex)
                                     + ".jpg"), composedImage)


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
    if imageType == "object":
        metafile = open(os.path.join(Object_DIR, imageType + "_flip.txt"), 'w+')
    elif imageType == "background":
        metafile = open(os.path.join(Background_DIR, imageType + "_flip.txt"), 'w+')

    for imageIndex, image in enumerate(images):
        imageClone = image.copy()
        imageFlipped = mirroring(imageClone)
        cv2.imwrite(os.path.join(saveDirectory, "flip/" + imageType + str(imageIndex)+ ".jpg"), imageFlipped)
        metafile.write("flip/" + imageType + str(imageIndex) + ".jpg" + "\t\t\t" + "has been fliped\n")
    metafile.close()
def augmentationScale(images, saveDirectory):
    metafile = open(os.path.join(Object_DIR, "object_flip.txt"), 'w+')
    for imageIndex, image in enumerate(images):
        for count, scaleRate in enumerate(ScaleSet):
            imageScaled = scale(image, scale_rate=scaleRate)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "scale/object" + str(imageIndex) +
                                     "Scale"+ str(count) + ".jpg"), imageScaled)
            metafile.write("scale/object" + str(imageIndex) + "Scale" + str(count) + ".jpg" + "\t\t\t" +
                           "%d has been scaled\n" % scaleRate)
    metafile.close()

def augmentationNoise(images, saveDirectory):
    metafile = open(os.path.abspath("../hashtag.txt"), 'w+')
    for imageIndex, image in enumerate(images):
        for count, noise in enumerate(NoiseSet):
            imageNoise = noisy(noise, image)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "noise/composed" + str(imageIndex) +
                                     noise + str(count) + ".jpg"), imageNoise)

def augmentationWeather(images, saveDirectory):
    metafile = open(os.path.join(ComposedImage_DIR, "composed_weather.txt"), 'w+')
    for imageIndex, image in enumerate(images):
        for count, weather in enumerate(WeatherSet):
            if weather == "rain":
                for rainEffect in range(1, 8):
                    imageRain = rainy(image, rainEffect, 0.6)
                    cv2.imwrite(os.path.join(saveDirectory,
                                             "weather/composed" + str(imageIndex) +
                                             "Rain" + str(count) + ".jpg"), imageRain)
                    metafile.write(
                        "brightness/" + "weather/composed" + str(imageIndex) + "Rain" + str(count) + ".jpg" + "\t\t\t" +
                        "%d has been rained\n" % rainEffect)
            elif weather == "fog":
                for fogEffect in range(1, 18):
                    imageFog = fog(image, fogEffect, 0.6)
                    cv2.imwrite(os.path.join(saveDirectory,
                                             "weather/composed" + str(imageIndex) +
                                             "Fog" + str(count) + ".jpg"), imageFog)
                    metafile.write(
                        "brightness/" + "weather/composed" + str(imageIndex) + "Fog" + str(count) + ".jpg" + "\t\t\t" +
                        "%d has been fogged\n" % fogEffect)
            elif weather == "snow":
                for snowEffect in range(1, 6):
                    imageSnow = snow(image, snowEffect, 0.6)
                    cv2.imwrite(os.path.join(saveDirectory,
                                             "weather/composed" + str(imageIndex) +
                                             "Snow" + str(count) + ".jpg"), imageSnow)
                    metafile.write(
                        "brightness/" + "weather/composed" + str(imageIndex) + "Snow" + str(count) + ".jpg" +
                        "\t\t\t" + "%d has been snowed\n" % snowEffect)
    metafile.close()

def augmentationBright(images, saveDirectory, imageType):
    if imageType == "object":
        metafile = open(os.path.join(Object_DIR, imageType + "_brightness.txt"), 'w+')
    elif imageType == "background":
        metafile = open(os.path.join(Background_DIR, imageType + "_brightness.txt"), 'w+')
    elif imageType == "composed":
        metafile = open(os.path.join(ComposedImage_DIR, imageType + "_brightness.txt"), 'w+')

    for imageIndex, image in enumerate(images):
        for count, bright in enumerate(BrightSet):
            imageBright = brightness_control(image, bright)
            cv2.imwrite(os.path.join(saveDirectory,
                                     "brightness/" + imageType + str(imageIndex) +
                                     "Bright" + str(count) + ".jpg"), imageBright)
            metafile.write("brightness/" + imageType + str(imageIndex) + "Bright" + str(count) + ".jpg" + "\t\t\t" +
                           "%d has been scaled\n" % bright)
    metafile.close()

if __name__ == '__main__':
    #model = model



    TestImages = loadImages(TEST_IMAGE_DIR)

    ObjectImageGenerate(TestImages, model)

    objectImages = loadImages(Object_DIR)

    backgroundImages = loadImages(Background_DIR)

    imageCompositeSample(objectImages, backgroundImages)

