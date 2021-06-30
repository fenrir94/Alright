#-*- coding:utf-8 -*-

import Core.imageGenerator as ImageGenerator
from Core.extractObject import *
from Core.augmentation import *
from Core.scaling import *
from Core.config import *
import datetime
from imgaug import augmenters as iaa

def augmentationBrightness(images, imageNames, label, dataFolder):
    brightnessRates = [-30, -60, -90, -120, 30, 60, 90, 120]
    for brightnessRate in brightnessRates:
        AugmentataionFolderName = "Brightness_" + str(brightnessRate) + "/"
        SAVE_DIRECTORY = IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName)
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder)
        createFolder(SAVE_DIRECTORY)
        object_discrimination = 0
        for imageIndex, image in enumerate(images):
                print("Image Brightness: ", imageIndex, ", BrightnessRate", brightnessRate)
                imageBright = brightness_control(image, int(brightnessRate), object_discrimination)
                # if (isValidImage(imageBright, image, 1, 1)):
                cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_Bright" + str(brightnessRate) + ".png", imageBright)

#  노이즈 종류
def augmentationNoise(images, imageNames, label, dataFolder):
    noiseKinds = ["gauss", "salt_pepper", "speckle"]
    for noiseKind in noiseKinds:
        SAVE_DIRECTORY = IMAGE_DIRECTORY + noiseKind + "/" + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + noiseKind + "/")
        createFolder(IMAGE_DIRECTORY + noiseKind + "/" + dataFolder)
        createFolder(SAVE_DIRECTORY)
        for imageIndex, image in enumerate(images):
                if noiseKind == "gauss":
                    for variance in GaussianVariances:
                        imageNoise = noisy(noiseKind, image, variance=variance)
                elif noiseKind == "salt_pepper":
                    for amount in SaltnPepperAmount:
                        imageNoise = noisy(noiseKind, image, amount=amount)
                else:
                    imageNoise = noisy(noiseKind, image)
                # if(isValidImage(imageNoise, image, 1, 1) == True):
                print("Noise: ", imageIndex, ", ", noiseKind)
                cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_" + noiseKind + ".png", imageNoise)

#  날씨 종류
def augmentationWeather(images, ImageNames, label, dataFolder, composite_rate):
    weatherKinds = ["rain", "snow", "fog"]
    for weatherKind in weatherKinds:
        SAVE_DIRECTORY = IMAGE_DIRECTORY + weatherKind + "_" + str(composite_rate) + "/" + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + weatherKind + "_" + str(composite_rate) + "/")
        createFolder(IMAGE_DIRECTORY + weatherKind + "_" + str(composite_rate) + "/" + dataFolder)
        createFolder(SAVE_DIRECTORY)

        effectNumber = 2
        for imageIndex, image in enumerate(images):
                if weatherKind == "rain":
                    imageRain = rainy(image, effectNumber)
                    print("Weather - Rain: Composed ", imageIndex, ", RainEffect")
                    # if (isValidImage(imageRain, image, 1, 1)):
                    cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_" + str(composite_rate) + "_RainEffect.png", imageRain)
                elif weatherKind == "fog":
                        imageFog = fog(image, effectNumber)
                        print("Weather - Fog: Composed ", imageIndex, ", FogEffect")
                        # if (isValidImage(imageFog, image, 1, 1)):
                        cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_" + str(composite_rate) + "_FogEffect.png",
                                    imageFog)
                elif weatherKind == "snow":
                        imageSnow = snow(image, effectNumber)
                        print("Weather - snow: Composed ", imageIndex, ", snowEffect")
                        # if (isValidImage(imageSnow, image, 1, 1)):
                        cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_" + str(composite_rate) + "_SnowEffect.png",
                                    imageSnow)



def imgaugBrightness(images, imageNames, label, dataFolder):
    brightnessRates = [-30, -60, -90, -120, 30, 60, 90, 120]
    for brightnessRate in brightnessRates:
        AugmentataionFolderName = "Brightness_imgaug_" + str(brightnessRate) + "/"
        SAVE_DIRECTORY = IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName)
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder)
        createFolder(SAVE_DIRECTORY)
        object_discrimination = 0
        imgaugBright = iaa.AddToBrightness()
        for imageIndex, image in enumerate(images):
            print("Image Brightness: ", imageIndex, ", BrightnessRate", brightnessRate)
            imageBright = imgaugBright(image=image)
            cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_Brightness_imgaug_" + str(brightnessRate) + ".png", imageBright)


def imgaugGaussian(images, imageNames, label, dataFolder):
    GaussianScales = [0.1, 0.2, 0.3, 0.4, 0.5]
    for GaussianScale in GaussianScales:
        AugmentataionFolderName = "Gausain_imgaug_" + str(GaussianScale) + "/"
        SAVE_DIRECTORY = IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName)
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder)
        createFolder(SAVE_DIRECTORY)
        imgaugGauss = iaa.AdditiveGaussianNoise(scale=GaussianScale*255)
        for imageIndex, image in enumerate(images):
            print("Image Noise Salt&Pepper: ", imageIndex, ", Gaussian_scale ", GaussianScale)
            image_Gaussian = imgaugGauss(image=image)
            cv2.imwrite( SAVE_DIRECTORY + imageNames[imageIndex] + "_Gaussian_imgaug_" + str(GaussianScale) + ".png", image_Gaussian)


def imgaugSaltandPepper(images, imageNames, label, dataFolder):
    SaltandPepperRates = [0.03, 0.06, 0.09, 0.12]
    for SaltandPepperRate in SaltandPepperRates:
        AugmentataionFolderName = "SaltandPepper_imgaug_" + str(SaltandPepperRate) + "/"
        SAVE_DIRECTORY = IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName)
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder)
        createFolder(SAVE_DIRECTORY)
        imgaugSaltandPepper = iaa.SaltAndPepper(SaltandPepperRate)
        for imageIndex, image in enumerate(images):
                print("Image Noise Salt&Pepper: ", imageIndex, ", SaltandPepperRate ", SaltandPepperRate)
                imageSaltandPepper = imgaugSaltandPepper(image=image)
                cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_SaltandPepper_imgaug_" + str(SaltandPepperRate) + ".png", imageSaltandPepper)


def imgaugCutout(images, imageNames, label, dataFolder):
    areaNumbers = [1, 2, 3, 4, 5]
    for areaNumber in areaNumbers:
        AugmentataionFolderName = "Cutout_imgaug_" + str(areaNumber) + "/"
        SAVE_DIRECTORY = IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName)
        createFolder(IMAGE_DIRECTORY + AugmentataionFolderName + dataFolder)
        createFolder(SAVE_DIRECTORY)
        imgaugCutout = iaa.Cutout(nb_iterations=areaNumber, size=0.1)
        for imageIndex, image in enumerate(images):
            print("Image  Cutout: ", imageIndex, ", Cutout_imgaug_", areaNumber)
            imageCutOut = imgaugCutout(image=image)
            cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_Cutout_imgaug_" + str(areaNumber) + ".png",
                        imageCutOut)


def imgaugWeather(images, imageNames, label, dataFolder):
    # weatherKinds = ["rain", "snow", "fog"]
    weatherKinds = ["rain", "fog"]
    # weatherKinds = ["snow"]
    for weatherKind in weatherKinds:
        SAVE_DIRECTORY = IMAGE_DIRECTORY + weatherKind + "_Imgaug" + "/" + dataFolder + label + "/"
        createFolder(IMAGE_DIRECTORY + weatherKind + "_imgaug/")
        createFolder(IMAGE_DIRECTORY + weatherKind + "_imgaug/" + dataFolder)
        createFolder(SAVE_DIRECTORY)

        imgaugRain = iaa.Rain(speed=(0.1, 0.3))
        imgaugFog = iaa.Fog()
        imgaugSnow = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))

        for imageIndex, image in enumerate(images):
            if weatherKind == "rain":
                imageRain = imgaugRain(image=image)
                print("Weather - Rain: Composed ", imageIndex, ", RainEffect")
                # if (isValidImage(imageRain, image, 1, 1)):
                cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_imgaug_Rain.png",
                            imageRain)
            elif weatherKind == "fog":
                imageFog = imgaugFog(image=image)
                print("Weather - Fog: Composed ", imageIndex, ", FogEffect")
                # if (isValidImage(imageFog, image, 1, 1)):
                cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_imgaug_Fog.png",
                            imageFog)
            elif weatherKind == "snow":
                imageSnow = imgaugSnow(image=image)
                print("Weather - snow: Composed ", imageIndex, ", snowEffect")
                # if (isValidImage(imageSnow, image, 1, 1)):
                cv2.imwrite(SAVE_DIRECTORY + imageNames[imageIndex] + "_imgaug_Snow.png",
                            imageSnow)




if __name__ == '__main__':

    timeformat = '%y/%m/%d %H:%M:%S'
    start = datetime.datetime.now().strftime(timeformat)
    start = datetime.datetime.strptime(start, timeformat)
    currenttime = datetime.datetime.now().strftime(timeformat)
    currenttime = datetime.datetime.strptime(currenttime, timeformat)
    print("Running Time: ", currenttime - start, "    Current Time: ", currenttime)

    FlipSet, ScaleSet, NoiseSet, WeatherSet, BrightnessSet = setParameters()


    # folderTestTrain = ["test/", "train/"]
    folderTestTrain = ["test/"]
    Labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
    # labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


    # for dataFolder in folderTestTrain:
    files = os.listdir(IMAGE_DIRECTORY + "test/")
    # labels = [file for file in files if not file.endswith(".bin") and not file.endswith(".py") and not file.endswith(".png")]
    for label in Labels:
        IMAGE_Folder_DIRECTORY = IMAGE_DIRECTORY + "test/" + label + "/"
        print(IMAGE_Folder_DIRECTORY)

        imageOrigin, imageNames = ImageGenerator.loadImages(IMAGE_Folder_DIRECTORY)
        # print(imageOrigin[0])

        # augmentationBrightness(imageOrigin, imageNames, label, "test/")
        # augmentationNoise(imageOrigin, imageNames, label, dataFolder)

        # imgaugCutout(imageOrigin, imageNames, label, "test/")
        # imgaugBrightness(imageOrigin, imageNames, label, "test/")
        imgaugGaussian(imageOrigin, imageNames, label, "test/")
        # imgaugSaltandPepper(imageOrigin, imageNames, label, "test/")
        # imgaugWeather(imageOrigin, imageNames, label, "test/")




