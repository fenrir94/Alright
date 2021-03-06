
import os
import time
import numpy
from Core.extractObject import *
from Core.augmentation import *
from Core.scaling import *
from Core.config import *
import xml.etree.ElementTree as elementtree
from Core.generateAnnotationXML import *
import random
import datetime

def loadImages(imageDirectory):
    files = os.listdir(imageDirectory)
    imagefiles = [file for file in files if file.endswith(".jpg")]
    images = []
    imagefileNames = []
    for filename in imagefiles:
        print(filename)
        image = cv2.imread(os.path.join(imageDirectory, filename))
        images.append(image)
        filename, _ = filename.split(".", 1)
        print(filename)
        imagefileNames.append(filename)

    return images, imagefileNames

def loadImagePaths(directory):
    files = os.listdir(directory)
    imagefiles = [file for file in files if file.endswith(".jpg")]
    imagefilePaths = []
    for filename in imagefiles:
        print(filename)
        imagefilePaths.append(filename)

    return imagefilePaths

def loadSubdirectoryImages(directory):
    images = []
    labels = []
    names = []
    for path, dir, files in os.walk(directory):
        for filename in files:
            # print(path, dir, filename)
            _, extend = filename.split(".", 1)
            if(extend != "jpg"):
                continue
            print(filename)
            image = cv2.imread(os.path.join(path, filename))
            images.append(image)
            name, label, _ = filename.split("_", 2)
            names.append(name)
            labels.append(label)

    return images, names, labels

# def loadbackgroundSubdirectoryImages(directory):
#     images = []
#     for path, dir, files in os.walk(directory):
#         for filename in files:
#             # print(path, dir, filename)
#             _, extend = filename.split(".", 1)
#             if(extend != "jpg"):
#                 continue
#             print(filename)
#             image = cv2.imread(os.path.join(path, filename))
#             images.append(image)
#
#     return images

def loadComposedImages(directory):
    images = []
    names = []
    files = os.listdir(directory)
    imagefiles = [file for file in files if file.endswith(".jpg")]
    for filename in imagefiles:
        annotationName, extend = filename.split(".", 1)
        image = cv2.imread(os.path.join(directory, filename))
        images.append(image)
        names.append(annotationName)
        print(filename)
        print(annotationName)
    return images, names

names_dict = {}
cnt = 0
f = open("C:/Users/user/Documents/GitHub/Alright/YOLOv3/data/voc.names", "r").readlines()
for line in f:
    line = line.strip()
    names_dict[line] = cnt
    cnt += 1

def randomCopy(lineNumber):
    testfile = open("C:/Users/user/Documents/GitHub/Alright/YOLOv3/data/my_data/testAugmented.txt", "r")
    randomfile = open("C:/Users/user/Documents/GitHub/Alright/YOLOv3/data/my_data/testAugmentedRandom" + str(lineNumber) + ".txt", "w")

    stack = []

    testlines = testfile.readlines()
    index = 0
    while index < lineNumber:
        line = random.randrange(0, 550815)
        if stack.count(line) == 0:
            stack.append(line)
        else:
            continue
        testline = testlines[line].split()
        testline[0] = str(index)
        randomline = " ".join(testline) + " \n"
        print(randomfile)
        randomfile.write(randomline)
        index = index + 1

    randomfile.close()
    testfile.close()

def parse_xml(xmlpath):
    tree = elementree.parse(xmlpath)

    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    objects = [width, height]

    for obj in tree.findall('object'):
        difficult = obj.find('difficult').text
        if difficult == '1':
            continue
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text
        name = str(names_dict[name])
        objects.extend([name, xmin, ymin, xmax, ymax])
    if len(objects) > 1:
        return objects
    else:
        return None

def generateTest():
    testfile = open("C:/Users/user/Documents/GitHub/Alright/YOLOv3/data/my_data/testNoBright.txt", "w")

    test_count = 0

    # testNames = loadImagePaths(TEST_IMAGE_DIRECTORY)
    # for imageName in testNames:
    #     name, _ = imageName.split(".", 1)
    #     xmlpath = Annotation_Test_DIRECTORY + name + ".xml"
    #     object = parse_xml(xmlpath)
    #     objects = ' '.join(object)
    #     print(str(test_count) + " " + TEST_IMAGE_DIRECTORY + imageName)
    #     testfile.write(str(test_count) + " " + TEST_IMAGE_DIRECTORY + imageName + " " + objects + " \n")
    #     test_count = test_count + 1
    # testNames.clear()

    composedNames = loadImagePaths(ComposedImage_DIRECTORY)
    for imageName in composedNames:
        name, _ = imageName.split(".", 1)
        xmlpath = Annotation_Composed_DIRECTORY + name + ".xml"
        object = parse_xml(xmlpath)
        objects = ' '.join(object)
        print(str(test_count) + " " + ComposedImage_DIRECTORY + imageName)
        testfile.write(str(test_count) + " " + ComposedImage_DIRECTORY + imageName + " " + objects + " \n")
        test_count = test_count + 1
    composedNames.clear()

    augmentedbrightnessNames = loadImagePaths(ComposedImage_Brightness_DIRECTORY)
    for imageName in augmentedbrightnessNames:
        number, label, count, _ = imageName.split("_", 3)
        xmlpath = Annotation_Composed_DIRECTORY + number + "_" + label + "_" + count + ".xml"
        object = parse_xml(xmlpath)
        objects = ' '.join(object)
        print(str(test_count) + " " + ComposedImage_Brightness_DIRECTORY + imageName)
        testfile.write(str(test_count) + " " + ComposedImage_Brightness_DIRECTORY + imageName + " " + objects + " \n")
        test_count = test_count + 1
    augmentedbrightnessNames.clear()

    augmentednoiseNames = loadImagePaths(ComposedImage_Noise_DIRECTORY)
    for imageName in augmentednoiseNames:
        number, label, count, _ = imageName.split("_", 3)
        xmlpath = Annotation_Composed_DIRECTORY + number + "_" + label + "_" + count + ".xml"
        object = parse_xml(xmlpath)
        objects = ' '.join(object)
        print(str(test_count) + " " + ComposedImage_Noise_DIRECTORY + imageName)
        testfile.write(str(test_count) + " " + ComposedImage_Noise_DIRECTORY + imageName + " " + objects + " \n")
        test_count = test_count + 1
    augmentednoiseNames.clear()

    augmentedweatherNames = loadImagePaths(ComposedImage_Weather_DIRECTORY)
    for imageName in augmentedweatherNames:
        number, label, count, _ = imageName.split("_", 3)
        xmlpath = Annotation_Composed_DIRECTORY + number + "_" + label + "_" + count + ".xml"
        object = parse_xml(xmlpath)
        objects = ' '.join(object)
        print(str(test_count) + " " + ComposedImage_Weather_DIRECTORY + imageName)
        testfile.write(str(test_count) + " " + ComposedImage_Weather_DIRECTORY + imageName + " " + objects + " \n")
        test_count = test_count + 1
    augmentedweatherNames.clear()

    testfile.close()

def getTestImagelabels(imagefileNames):
    testlabels = []
    for index, imagefileName in enumerate(imagefileNames):
        xmlpath = Annotation_Test_DIRECTORY + imagefileName + ".xml"
        labelTree = elementtree.parse(xmlpath)
        obj = labelTree.find('object')
        label = obj.find('name').text
        print(label)
        testlabels.append(label)

    return testlabels

def backgroundImageScaling(backgrounds, width, height, directory):
    for backgroundIndex, background in enumerate(backgrounds):
        background_height, background_width, _ = background.shape
        if height < background_height:
            rate = height / background_height
            background = scale(background, rate)
        if width < background_width:
            rate = height/ background_width
            background = scale(background, rate)
        print(background_height)
        print(background_width)
        cv2.imwrite(directory + "background" + str(backgroundIndex) + ".jpg", background)
        print(directory + "background" + str(backgroundIndex) + ".jpg")

def ObjectImageGenerate(images, model, names, labels, save_directory):
    for imageindex, image in enumerate(images):
        results = model.detect([image], verbose=1)

        r = results[0]
        class_id = r['class_ids']
        label = class_names[class_id[0]]
        print(label)
        roi = extractObject(image, r['rois'], r['masks'], r['class_ids'])
        image = object_distinction(roi)
        cv2.imwrite(save_directory + names[imageindex] + "_" + labels[imageindex]+ "_" + str(imageindex) + ".jpg", image)

        print("imageGenerated:     " + save_directory + names[imageindex]+ "_" + labels[imageindex]+ "_" + str(imageindex) + ".jpg")


def imageComposite(objects, backgrounds, save_directory, names, labels):

    composedCount = 0
    for objectIndex, object in enumerate(objects):
        for backgroundIndex, background in enumerate(backgrounds):
            objectClone = object.copy()
            backgroundClone = background.copy()
            rate = 1
            object_height, object_width, _ = objectClone.shape
            background_height, background_width, _ = background.shape
            while (object_height >= background_height or object_width >= background_width):
                if object_height >= background_height:
                    rate = background_height / (3*object_height)
                    objectClone = scale(objectClone, rate)
                elif object_width >= background_width:
                    rate = background_width/(3*object_width)
                    objectClone = scale(objectClone, rate)
                print(rate)
                object_height, object_width, _ = objectClone.shape
                print(background_height, "   ", object_height)
                print(background_width, "   ", object_width)
            x, y = random_location(background_width, background_height, object_width, object_height)
            composedImage = attachImage(backgroundClone, objectClone, x, y)

            annotationName = names[objectIndex] + "_" + labels[objectIndex] + "_" + str(composedCount)
            imageName = annotationName + "_" + "object" + str(objectIndex) +"background" + str(backgroundIndex)
            print("Image Composed: Object ", objectIndex, " in Background ", backgroundIndex)
            print(save_directory + annotationName + ".jpg")
            cv2.imwrite(save_directory + annotationName + ".jpg", composedImage)
            # print("Image saved: " + annotationName)
            generateAnnoationXML(annotationName, labels[objectIndex], x, y, object_height,
                                 object_width, background_height,background_width,
                                 TEST_IMAGE_DIRECTORY, Annotation_Composed_DIRECTORY)

            composedCount = composedCount + 1
            print("xml created   ", save_directory, "   ", annotationName)

def objectAugmentation(objects, names, labels, doAugmentation):
    if doAugmentation == "True":
        print("Object Augmentation Started")
        augmentationFlip(objects, Object_Flip_DIRECTORY, "object", names, labels, doAugmentation=doObjectFlip)
        augmentationScale(objects, Object_Scale_DIRECTORY, names, labels, doAugmenation=doObjectScale, )
        augmentationBright(objects, Object_Brightness_DIRECTORY, "object", names, labels, doAugmentation=doObjectBrightness)
        print("Object Augmentation Finished")
    else:
        print("Do Not Object Augmentation")

def backgroundAugmentation(backgrounds, doAugmenation):
    if doAugmenation == "True":
        print("Background Augmentation Started")
        augmentationFlip(backgrounds, Background_Flip_DIRECTORY, "background", names=None, labels= None, doAugmentation=doBackgroundFlip)
        augmentationBright(backgrounds, Background_Brightness_DIRECTORY, "background", names=None, labels= None, doAugmentation=doBackgroundBrightness)
        print("Background Augmentation Finished")
    else:
        print("Do Not Background Augmentation")

def composedAugmentation(composed, names, doAugmnetation):
    if doAugmnetation == "True":
        print("Composed Augmentation Started")

        augmentationWeather(composed, ComposedImage_Weather_DIRECTORY, names, doAugmentation=doComposedWeather)
        augmentationBright(composed, ComposedImage_Brightness_DIRECTORY, "composed", names, labels=None, doAugmentation=doComposedBrightness)
        augmentationNoise(composed, ComposedImage_Noise_DIRECTORY, names, doAugmentation=doComposedNoise)
        print("Composed Augmentation Finish")

    else:
        print("Do Not Composed Image Augmentation")


def augmentationFlip(images, saveDirectory, imageType, names, labels, doAugmentation):
    if doAugmentation == "True":
        if imageType == "object":
            metafile = open(os.path.join(Object_DIRECTORY, imageType + "_flip.txt"), 'w+')
        elif imageType == "background":
            metafile = open(os.path.join(Background_DIRECTORY, imageType + "_flip.txt"), 'w+')

        for imageIndex, image in enumerate(images):
            imageClone = image.copy()
            imageFlipped = mirroring(imageClone)
            print("Image Flipped: ", imageType, imageIndex)
            if imageType == "object":
                cv2.imwrite(saveDirectory + names[imageIndex]+"_"+labels[imageIndex]+"_"+str(imageIndex)+".jpg", imageFlipped)
            else:
                cv2.imwrite(saveDirectory + imageType + str(imageIndex)+".jpg", imageFlipped)
            metafile.write("flip/" + imageType + str(imageIndex) + ".jpg" + "\t\t\t" + "has been fliped\n")
        metafile.close()
        print("Flip Finished")
    else:
        print("No Not Flip")

def augmentationScale(images, saveDirectory, names, labels, doAugmenation):
    if doAugmenation == "True":
        metafile = open(os.path.join(Object_DIRECTORY, "object_scale.txt"), 'w+')
        for imageIndex, image in enumerate(images):
            for count, scaleRate in enumerate(ScaleSet):
                imageScaled = scale(image, scale_rate=float(scaleRate))
                print("Image Scaling: Object ", imageIndex, "to Scale Rate ", scaleRate)
                cv2.imwrite(saveDirectory + names[imageIndex] + "_"+labels[imageIndex]+ "_" + "object" + str(imageIndex) +
                                        "Scale"+ str(count) + ".jpg", imageScaled)
                metafile.write("scale/object" + str(imageIndex) + "Scale" + str(count) + ".jpg" + "\t\t\t" +
                               "%s has been scaled\n" % scaleRate)
        metafile.close()
        print("Scale Finished")
    else:
        print("Do Not Scale")

def augmentationNoise(images, saveDirectory, names, doAugmentation):
    if doAugmentation == "True":
        metafile = open(os.path.join(ComposedImage_DIRECTORY, "composed_noise.txt"), 'w+')
        for imageIndex, image in enumerate(images):
            for count, noise in enumerate(NoiseSet):
                print("Noise: ", imageIndex, ", ", noise)
                imageNoise = noisy(noise, image)
                cv2.imwrite(saveDirectory + names[imageIndex] + "_" + noise + str(count) + ".jpg", imageNoise)
                metafile.write("noise/composed" + str(imageIndex) + noise + str(count) + ".jpg" + "\t\t\t" +
                               "%s has been noised\n" % noise)
        metafile.close()
        print("Noise Finished")
    else:
        print("Do Not Noise")

def augmentationWeather(images, saveDirectory, names, doAugmentation):
    if doAugmentation == "True":
        metafile = open(os.path.join(ComposedImage_DIRECTORY, "composed_weather.txt"), 'w+')
        for imageIndex, image in enumerate(images):
            for count, weather in enumerate(WeatherSet):
                if weather == "rain":
                    for rainEffect in range(1, 3):
                        imageRain = rainy(image, rainEffect, 0.6)
                        print("Weather - Rain: Composed ", imageIndex, ", RainEffect ", rainEffect)
                        cv2.imwrite(saveDirectory + names[imageIndex] + "_RainEffect" + str(rainEffect) + ".jpg", imageRain)
                        metafile.write(
                            "brightness/" + "weather/composed" + str(imageIndex) + "Rain" + str(count) + "effect" +
                            str(rainEffect) + ".jpg" + "\t\t\t" + "%s has been rained\n" % rainEffect)
                elif weather == "fog":
                    for fogEffect in range(2, 4):
                        imageFog = fog(image, fogEffect, 0.6)
                        print("Weather - Fog: Composed ", imageIndex, ", FogEffect ", fogEffect)
                        cv2.imwrite(saveDirectory + names[imageIndex] + "_FogEffect" + str(fogEffect) + ".jpg", imageFog)
                        metafile.write(
                            "brightness/" + "weather/composed" + str(imageIndex) + "Fog" + str(count) + "effect"
                            + str(fogEffect) + ".jpg" + "\t\t\t" + "%s has been fogged\n" % fogEffect)
                elif weather == "snow":
                    for snowEffect in range(2, 4):
                        imageSnow = snow(image, snowEffect, 0.6)
                        print("Weather - snow: Composed ", imageIndex, ", snowEffect ", snowEffect)
                        cv2.imwrite(saveDirectory + names[imageIndex] + "_SnowEffect" +str(snowEffect) + ".jpg", imageSnow)
                        metafile.write(
                            "brightness/" + "weather/composed" + str(imageIndex) + "Snow" + str(count) + "effect"
                            + str(snowEffect) + ".jpg" + "\t\t\t" + "%s has been snowed\n" % snowEffect)
        metafile.close()
        print("Weather Finished")
    else:
        print("Do Not Weather")

def augmentationBright(images, saveDirectory, imageType, names, labels, doAugmentation):
    if doAugmentation == "True":
        object_discrimination = 0
        if imageType == "object":
            metafile = open(os.path.join(Object_DIRECTORY, imageType + "_brightness.txt"), 'w+')
            object_discrimination = 35
        elif imageType == "background":
            metafile = open(os.path.join(Background_DIRECTORY, imageType + "_brightness.txt"), 'w+')
        elif imageType == "composed":
            metafile = open(os.path.join(ComposedImage_DIRECTORY, imageType + "_brightness.txt"), 'w+')

        for imageIndex, image in enumerate(images):
            for count, bright in enumerate(BrightnessSet):
                print("Image Brightness: ", imageType, imageIndex, ", BrightnessRate", bright)
                imageBright = brightness_control(image, int(bright), object_discrimination)
                if imageType == "background":
                    cv2.imwrite(saveDirectory + imageType + str(imageIndex) + "Bright" + str(count) + ".jpg", imageBright)
                elif imageType == "composed":
                    cv2.imwrite(saveDirectory + names[imageIndex] + "_" + imageType + "Bright" + str(count) + ".jpg", imageBright)
                else:
                    cv2.imwrite(saveDirectory + names[imageIndex] + "_" + labels[imageIndex] +
                                             "_" + imageType + str(imageIndex) + "Bright" + str(count) + ".jpg", imageBright)
                metafile.write("brightness/" + imageType + str(imageIndex) + "Bright" + str(count) + ".jpg" + "\t\t\t" +

                               "%s has been scaled\n" % bright)
        metafile.close()
        print("Brightness Finished")
    else:
        print("Do Not Brightness")



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

    timeformat = '%y/%m/%d %H:%M:%S'
    start = datetime.datetime.now().strftime(timeformat)
    start = datetime.datetime.strptime(start, timeformat)
    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

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

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    print(TEST_IMAGE_DIRECTORY)
    print(Object_DIRECTORY)
    print(Background_DIRECTORY)
    print(ComposedImage_DIRECTORY)

    generateDirectories()

    TestImages, ImageNames = loadImages(TEST_IMAGE_DIRECTORY)

    Labels = getTestImagelabels(ImageNames)

    ObjectImageGenerate(TestImages, model, ImageNames, Labels, Object_DIRECTORY)

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    objectImages, _ = loadImages(Object_DIRECTORY)
    objectAugmentation(objectImages, ImageNames, Labels, doAugmentation=doObjectAugmentation)

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    backgroundImages, _ = loadImages(Background_DIRECTORY)

    backgroundImageScaling(backgroundImages, 416, 416, Background_Scale_DIRECTORY)

    backgroundImages.clear()

    backgroundImages, _ = loadImages(Background_Scale_DIRECTORY)

    backgroundAugmentation(backgroundImages, doAugmenation=doBackgroundAugmentation)

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    additionalObjectImage, additionalObjectName, additionalObjectLabel = loadSubdirectoryImages(Object_Augmented_DIRECTORY)

    objectImages.extend(additionalObjectImage)
    ImageNames.extend(additionalObjectName)
    Labels.extend(additionalObjectLabel)

    BackgroundBrightness, _ = loadImages(Background_Brightness_DIRECTORY)
    BackgroundFlip, _ = loadImages(Background_Flip_DIRECTORY)

    backgroundImages.extend(BackgroundBrightness)
    backgroundImages.extend(BackgroundFlip)

    imageComposite(objectImages, backgroundImages, ComposedImage_DIRECTORY, ImageNames, Labels)

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    objectImages.clear()
    backgroundImages.clear()

    composedImages, composedImagesNames = loadComposedImages(ComposedImage_DIRECTORY)

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    composedAugmentation(composedImages, composedImagesNames, doAugmnetation=doComposedImageAugmentation)

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    composedImages.clear()

    generateTest()

    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime, "   Stated Time: ", start)

    # randomCopy(4000)
    #
    # print("Running Time: ", time.time() - start, "    Current Time: ", time.time())

    print("Finish")
