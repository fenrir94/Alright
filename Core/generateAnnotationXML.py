from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
# import xml.dom.minidom as minidom

def indent(elem, level=0):
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def generateAnnoationXML(filename, label, _x, _y, boxHeight, boxWidth, _height, _width, test_image_directory, save_directory):
    imagefilename = str(filename) + '.jpg'


    root = Element("annotation")
    tree = ElementTree(root)

    folder = SubElement(root,"folder")
    folder.text = test_image_directory

    fileid = SubElement(root, "filename")
    fileid.text = imagefilename
    # print(fileid.text)

    source = SubElement(root, "source")

    database = SubElement(source, "database")
    database.text = "Augmented"
    # print(database.text)
    annoatation = SubElement(source, "annoatation")
    annoatation.text = "PASCAL VOC"
    # print(annoatation.text)
    image = SubElement(source, "image")
    image.text = "?"
    # print(image.text)
    flickrid = SubElement(source, "flickrid")
    flickrid.text = "?"
    # print(flickrid.text)

    owner = SubElement(root, "owner")

    flickrid = SubElement(owner, "flickrid")
    flickrid.text = "?"
    name = SubElement(owner, "name")
    name.text = "?"

    size = SubElement(root, "size")

    width = SubElement(size, "width")
    width.text = str(_width)
    height = SubElement(size, "height")
    height.text = str(_height)
    depth = SubElement(size, "depth")
    depth.text = "3"

    segmented = SubElement(root, "segmented")
    segmented.text = "0"

    objectInstance = SubElement(root, "object")

    name = SubElement(objectInstance, "name")
    name.text = label
    pose = SubElement(objectInstance, "pose")
    pose.text = "Unspeicified"
    truncated = SubElement(objectInstance, "truncated")
    truncated.text = "0"
    difficult = SubElement(objectInstance, "difficult")
    difficult.text = "0"

    bndbox = SubElement(objectInstance, "bndbox")

    xmin = SubElement(bndbox, "xmin")
    xmin.text = str(_x)
    ymin = SubElement(bndbox, "ymin")
    ymin.text = str(_y)
    xmax = SubElement(bndbox, "xmax")
    xmax.text = str(boxWidth + _x)
    ymax = SubElement(bndbox, "ymax")
    ymax.text = str(boxHeight + _y)

    indent(root)

    annoatationName = str(save_directory) + str(filename) + ".xml"
    ElementTree(root).write(annoatationName, "utf-8")
    # tree.write(open("test.xml", 'w'), encoding="unicode")
    # tree.write(open(ElementTree.tostring(save_directory + id+'.xml'), 'w'), encoding='unicode')
    # print("xml created")


    # return tree
#
# if __name__=='__main__':
#     root = generateAnnoationXML(11, "dog", 10, 20, 50, 60, 200, 400, "testimages", "testimages")
