import os
import sys
import math
import requests
import time
import configparser
from multiprocessing import Process, Manager, freeze_support
from bs4 import BeautifulSoup
# from .config import Background_DIR


def get_elapsed_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds


def get_max_page_count(url, keyword):
    main_page_src = requests.get(f'{url}{keyword}/?page=1')
    plain_main_page_text = main_page_src.text
    soup_main_page = BeautifulSoup(plain_main_page_text, 'lxml')
    # if parsing is not working try checking below
    max_page_str = soup_main_page.find('form', {'class': 'add_search_params pure-form hide-xs hide-sm hide-md'})
    max_page = None
    for s in max_page_str.get_text().split():
        if s.isdigit():
            max_page = int(s)

    if max_page is None:
        sys.exit('max_page parsing is not working! try checking get_max_page_count html parsing source code')

    return max_page


def execute_multiple_process(keyword, start_page, max_page, url, cur_dir, hash_tag_list,
                             hash_tag_contents):
    process1 = Process(target=get_image_data, args=(keyword, int(start_page), math.floor(int(max_page / 4)), url,
                                                    cur_dir, hash_tag_list, hash_tag_contents))
    print(f"1st start_page : {start_page}, max_page : {math.floor(max_page / 4)}")
    start_page = math.floor((max_page / 4) + 1)
    process2 = Process(target=get_image_data, args=(keyword, int(start_page), math.floor(int(max_page / 2)), url,
                                                    cur_dir, hash_tag_list, hash_tag_contents))
    print(f"2nd start_page : {start_page}, max_page : {math.floor(max_page / 2)}")
    start_page = math.floor((max_page / 2) + 1)
    process3 = Process(target=get_image_data, args=(keyword, int(start_page), math.floor(int(max_page - (max_page / 4)))
                                                    , url, cur_dir, hash_tag_list, hash_tag_contents))
    print(f"3rd start_page : {start_page}, max_page : {math.floor(max_page - (max_page / 4))}")
    start_page = math.floor(start_page + (max_page / 4))
    process4 = Process(target=get_image_data, args=(keyword, int(start_page), math.floor(int(max_page)), url, cur_dir,
                                                    hash_tag_list, hash_tag_contents))
    print(f"4th start_page : {start_page}, max_page : {max_page}")
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()


def get_image_data(keyword, start_page, max_page, url, cur_dir, hash_tag_list, hash_tag_content_list):
    for i in range(start_page, max_page + 1):
        image_file_name_and_hash_tag = {}
        cur_url = f'{url}{keyword}/?pagi={i}'
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        page_src = requests.get(cur_url)
        plain_page_text = page_src.text
        soup_page = BeautifulSoup(plain_page_text, 'lxml')
        for img in soup_page.find_all('img'):
            if "https://cdn.pixabay.com/photo/" in img.get('src'):
                response = requests.get(img.get('src')).content
                hash_tag = str(img.get('alt')).lower()
                image_file_name = f"{img.get('src')[47:-4]}${hash_tag}{img.get('src')[-4:]}"
                img_file = open(cur_dir + '\\' + image_file_name, 'wb')
                img_file.write(response)
                img_file.close()
                hash_tag_list.append(hash_tag.split(', '))
                image_file_name_and_hash_tag.update({f"{img.get('src')[47:]}": hash_tag})
            elif "/static/img/blank.gif" in img.get('src') and "https://cdn.pixabay.com/photo/" in img.get('data-lazy'):
                response = requests.get(img.get('data-lazy')).content
                hash_tag = str(img.get('alt')).lower()
                image_file_name = f"{img.get('data-lazy')[47:-4]}${hash_tag}{img.get('data-lazy')[-4:]}"
                img_file = open(cur_dir + '\\' + image_file_name, 'wb')
                img_file.write(response)
                img_file.close()
                hash_tag_list.append(hash_tag.split(', '))
                image_file_name_and_hash_tag.update({f"{img.get('data-lazy')[47:]}": hash_tag})
        hash_tag_content_list[i] = image_file_name_and_hash_tag


def count_frequency(param_list):
    frequency = {}
    for items in param_list:
        for item in items:
            if item in frequency:
                frequency[item] += 1
            else:
                frequency[item] = 1

    return frequency


def dict_to_csv(param_dict):
    csv_columns = ['단어', '빈도 수']
    csv_file = os.path.abspath("../hash_tag_Counting.csv")
    try:
        with open(csv_file, 'w', encoding='utf-8-sig') as csvfile:
            csvfile.write(f"{csv_columns[0]},{csv_columns[1]}\n")
            for data in param_dict.keys():
                csvfile.write(f"{data},{param_dict[data]}\n")
    except IOError:
        print("I/O error")


def extract_hash_tags():
    config = configparser.ConfigParser()
    config.read('../config.ini')
    keyword = config['CRAWLING']['EXTRACT_HASH_TAGS_CRAWLED_KEYWORD']
    keyword = keyword.lower()
    cur_dir = os.path.abspath(f"../images/background/crawledImages/{keyword}")
    list_dict = {}
    for list in os.listdir(cur_dir):
        list_tmp = list.split('$')
        list_tmp[1] = list_tmp[1][:-4]  # delete .jpg, .png, ...
        list_dict.update({f"{list_tmp[0]}": f"{list_tmp[1]}"})

    print(list_dict)
    return list_dict





def pixabay_crawling_images(keyword):
    start = time.time()
    keyword = keyword.lower()
    freeze_support()  # window dependency error fix
    manager = Manager()  # multiprocessing manager
    hash_tag_list = manager.list()
    hash_tag_contents = manager.dict()
    Background_DIR = os.path.abspath("../images/background/crawledImages")
    url = 'https://pixabay.com/en/images/search/'
    cur_dir = f'{Background_DIR}/{keyword}/'
    config = configparser.ConfigParser()
    config.read('../config.ini')
    start_page = 1
    if config['CRAWLING']['MAX_CRAWLING_IMAGE_PAGE'] == 'Max':
        max_page = get_max_page_count(url, keyword)
    else:
        max_page = int(config['CRAWLING']['MAX_CRAWLING_IMAGE_PAGE'])
    execute_multiple_process(keyword, start_page, max_page, url, cur_dir, hash_tag_list,
                             hash_tag_contents)
    hash_tag_content_sorted = dict(sorted(hash_tag_contents.items(), key=(lambda x: x[0]), reverse=False))
    images_number = 0
    meta_file = open(os.path.abspath("../hash_tag.txt"),  'w+')
    for hash_tag_content in hash_tag_content_sorted.keys():
        meta_file.write("%d/%d page\n" % (hash_tag_content, max_page))
        for image_file_name, hash_tag in hash_tag_contents[hash_tag_content].items():
            meta_file.write(image_file_name + "\t\t\t" + hash_tag + "\n")
            images_number += 1
    processed_before_hash_tag = time.time()
    before_execute_hash_tag_hours, before_execute_hash_tag_minutes, before_execute_hash_tag_seconds = get_elapsed_time(
        start, processed_before_hash_tag)
    meta_file.write(f"Total {images_number} image files with spend time ")
    meta_file.write("{:0>2}:{:0>2}:{:05.2f}".format(int(before_execute_hash_tag_hours),
                                                    int(before_execute_hash_tag_minutes),
                                                    before_execute_hash_tag_seconds))
    meta_file.close()
    hash_tag_frequency = count_frequency(hash_tag_list)
    hash_tag_sorted = dict(sorted(hash_tag_frequency.items(), key=(lambda x: x[1]), reverse=True))
    dict_to_csv(hash_tag_sorted)
    end = time.time()
    hours, minutes, seconds = get_elapsed_time(start, end)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
