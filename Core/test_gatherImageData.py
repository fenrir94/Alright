import configparser
from Core.gatherImageData import pixabay_crawling_images, extract_hash_tags


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('../config.ini')

    # keyword convert to lower characters
    pixabay_crawling_images(config['CRAWLING']['SEARCH_KEYWORD'])
    # extract_hash_tags()
