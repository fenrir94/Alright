from Core.gatherImageData import crawlingImages


def testCrawlingPixabey(keyword):
    crawlingImages(keyword)


if __name__ == '__main__':
    testCrawlingPixabey("도로")