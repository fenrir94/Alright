import os
import requests
import time
from multiprocessing import Process, Manager, freeze_support
from bs4 import BeautifulSoup

def getImageDatas(keyword, start_page, max_page, url, cur_dir, hashtag_list, imagefile_name_list, hashtag_content_list):
    for i in range(start_page, max_page + 1):
        cur_url = url + keyword + '/?pagi=' + str(i)
        page_src = requests.get(cur_url)
        plain_page_text = page_src.text
        soup_page = BeautifulSoup(plain_page_text, 'lxml')
        for img in soup_page.find_all('img'):
            if "https://cdn.pixabay.com/photo/" in img.get('src'):
                response = requests.get(img.get('src')).content
                hashtag = img.get('alt')
                imagefile_name = img.get('src')[47:]
                # hashtagfile_name = imagefile_name[:imagefile_name.rfind('.')] + '.txt'
                if not os.path.exists(cur_dir):
                    os.makedirs(cur_dir)
                img_file = open(cur_dir + '\\' + imagefile_name, 'wb')
                # metafile = open(cur_dir + '\\' + hashtagfile_name, 'w')
                img_file.write(response)
                img_file.close()
                hashtag_list.append(hashtag.split(', '))
                imagefile_name_list.append(imagefile_name)
                hashtag_content_list.append(hashtag)

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


if __name__ == '__main__':
    start = time.time()
    freeze_support()
    global manager
    manager = Manager()
    global hashtag_list
    global imagefile_name
    global hashtag_content
    hashtag_list = manager.list()
    imagefile_name = manager.list()
    hashtag_content = manager.list()
    url = 'https://pixabay.com/ko/images/search/'
    cur_dir = os.path.abspath("../images/background/" + "도로" + "/")
    main_page_src = requests.get(url + "도로" + '/?pagi=1')
    plain_main_page_text = main_page_src.text
    soup_main_page = BeautifulSoup(plain_main_page_text, 'lxml')
    max_page_str = soup_main_page.find('form', {'class': 'add_search_params pure-form hide-xs hide-sm hide-md'})
    for s in max_page_str.get_text().split():
        if s.isdigit():
            max_page = 5
    start_page = 1
    process1 = Process(target=getImageDatas, args=("도로", int(start_page), int(max_page / 4), url, cur_dir, hashtag_list, imagefile_name, hashtag_content))
    print("1st start_page : %d, max_page : %d" % (start_page, max_page / 4))
    start_page = (max_page / 4) + 1
    process2 = Process(target=getImageDatas, args=("도로", int(start_page), int(max_page / 2), url, cur_dir, hashtag_list, imagefile_name, hashtag_content))
    print("2nd start_page : %d, max_page : %d" % (start_page, max_page / 2))
    start_page = (max_page / 2) + 1
    process3 = Process(target=getImageDatas, args=("도로", int(start_page), int(max_page - (max_page / 4)), url, cur_dir, hashtag_list, imagefile_name, hashtag_content))
    print("3rd start_page : %d, max_page : %d" % (start_page, max_page - (max_page / 4)))
    start_page = start_page + (max_page / 4)
    process4 = Process(target=getImageDatas, args=("도로", int(start_page), int(max_page), url, cur_dir, hashtag_list, imagefile_name, hashtag_content))
    print("4th start_page : %d, max_page : %d" % (start_page, max_page))
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()
    metafile = open(os.path.abspath("../hashtag.txt"),  'w+')
    for i in range(0, len(imagefile_name)):
        metafile.write(imagefile_name[i] + "\t\t\t" + hashtag_content[i] + "\n")
    metafile.close()
    hashtag_frequency = CountFrequency(hashtag_list)
    hashtag_sorted = dict(sorted(hashtag_frequency.items(), key=(lambda x: x[1]), reverse=True))
    dictToCSV(hashtag_sorted)
    # getImageDatas("도로")
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))