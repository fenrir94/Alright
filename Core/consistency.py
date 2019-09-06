import configparser


def is_consist(background_keyword, test_image_hash_tag):
    config = configparser.ConfigParser()
    config.read('../config.ini')

    matching_list = config['CONSISTENCY'][background_keyword].replace('\n', ' ').split(', ')

    result = False
    for tag in matching_list:
        if tag == test_image_hash_tag:
            result = True

    # print(matching_list)

    return result
