import os
import requests
import time
from multiprocessing import Process, Manager, freeze_support
from bs4 import BeautifulSoup


def getImageDatas(keyword, start_page, max_page, url, cur_dir, hashtag_list, hashtag_content_list):
    for i in range(start_page, max_page + 1):
        imagefile_name_and_hashtag = {}
        cur_url = url + keyword + '/?pagi=' + str(i)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        page_src = requests.get(cur_url)
        plain_page_text = page_src.text
        soup_page = BeautifulSoup(plain_page_text, 'lxml')
        for img in soup_page.find_all('img'):
            if "https://cdn.pixabay.com/photo/" in img.get('src'):
                response = requests.get(img.get('src')).content
                hashtag = img.get('alt')
                imagefile_name = img.get('src')[47:]
                img_file = open(cur_dir + '\\' + imagefile_name, 'wb')
                img_file.write(response)
                img_file.close()
                hashtag_list.append(hashtag.split(', '))
                imagefile_name_and_hashtag.update({imagefile_name: hashtag})
            elif "/static/img/blank.gif" in img.get('src') and "https://cdn.pixabay.com/photo/" in img.get('data-lazy'):
                    response = requests.get(img.get('data-lazy')).content
                    hashtag = img.get('alt')
                    imagefile_name = img.get('data-lazy')[47:]
                    img_file = open(cur_dir + '\\' + imagefile_name, 'wb')
                    img_file.write(response)
                    img_file.close()
                    hashtag_list.append(hashtag.split(', '))
                    imagefile_name_and_hashtag.update({imagefile_name: hashtag})
        hashtag_content_list[i] = imagefile_name_and_hashtag



def CountFrequency(param_list):
    freq = {}
    for items in param_list:
        for item in items:
            if (item in freq):
                freq[item] += 1
            else:
                freq[item] = 1

    return freq

def dictToCSV(param_dict):
    csv_columns = ['단어', '빈도 수']
    csv_file = os.path.abspath("../hashtagCounting.csv")
    try:
        with open(csv_file, 'w', encoding='utf-8-sig') as csvfile:
            csvfile.write("%s,%s\n"%(csv_columns[0], csv_columns[1]))
            for data in param_dict.keys():
                csvfile.write("%s,%s\n"%(data, param_dict[data]))
    except IOError:
        print("I/O error")


def crawlingImages(keyword):
    start = time.time()
    freeze_support()
    global manager
    manager = Manager()
    global hashtag_list
    global hashtag_contents
    hashtag_list = manager.list()
    hashtag_contents = manager.dict()
    url = 'https://pixabay.com/ko/images/search/'
    cur_dir = os.path.abspath("../images/background/" + keyword + "/")
    main_page_src = requests.get(url + keyword + '/?pagi=1')
    plain_main_page_text = main_page_src.text
    soup_main_page = BeautifulSoup(plain_main_page_text, 'lxml')
    max_page_str = soup_main_page.find('form', {'class': 'add_search_params pure-form hide-xs hide-sm hide-md'})
    for s in max_page_str.get_text().split():
        if s.isdigit():
            # max_page = 5
            max_page = int(s)
    start_page = 1
    # getImageDatas(keyword, start_page, 2, url, cur_dir, hashtag_list, imagefile_name, hashtag_content)
    process1 = Process(target=getImageDatas, args=(keyword, int(start_page), int(max_page / 4), url, cur_dir, hashtag_list, hashtag_contents))
    print("1st start_page : %d, max_page : %d" % (start_page, max_page / 4))
    start_page = (max_page / 4) + 1
    process2 = Process(target=getImageDatas, args=(keyword, int(start_page), int(max_page / 2), url, cur_dir, hashtag_list, hashtag_contents))
    print("2nd start_page : %d, max_page : %d" % (start_page, max_page / 2))
    start_page = (max_page / 2) + 1
    process3 = Process(target=getImageDatas, args=(keyword, int(start_page), int(max_page - (max_page / 4)), url, cur_dir, hashtag_list, hashtag_contents))
    print("3rd start_page : %d, max_page : %d" % (start_page, max_page - (max_page / 4)))
    start_page = start_page + (max_page / 4)
    process4 = Process(target=getImageDatas, args=(keyword, int(start_page), int(max_page), url, cur_dir, hashtag_list, hashtag_contents))
    print("4th start_page : %d, max_page : %d" % (start_page, max_page))
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()
    hashtag_content_sorted =  dict(sorted(hashtag_contents.items(), key=(lambda x: x[0]), reverse=False))
    images_number = 0
    metafile = open(os.path.abspath("../hashtag.txt"),  'w+')
    for hashtag_content in hashtag_content_sorted.keys():
        metafile.write("%d/%d page\n" % (hashtag_content, max_page))
        for imagefile_name, hashtag in hashtag_contents[hashtag_content].items():
            metafile.write(imagefile_name + "\t\t\t" + hashtag + "\n")
            images_number += 1
    processed_before_hashtag = time.time()
    before_hashtag_hours, before_hashtag_rem = divmod(processed_before_hashtag - start, 3600)
    before_hashtag_minutes, before_hashtag_seconds = divmod(before_hashtag_rem, 60)
    metafile.write("Total %d imagefiles with spend time " %(images_number))
    metafile.write("{:0>2}:{:0>2}:{:05.2f}".format(int(before_hashtag_hours),int(before_hashtag_minutes), before_hashtag_seconds))
    metafile.close()
    hashtag_frequency = CountFrequency(hashtag_list)
    hashtag_sorted = dict(sorted(hashtag_frequency.items(), key=(lambda x: x[1]), reverse=True))
    dictToCSV(hashtag_sorted)
    # getImageDatas("도로")
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))