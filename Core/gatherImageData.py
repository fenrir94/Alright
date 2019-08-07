import os
import requests
import urllib.request
from bs4 import BeautifulSoup

def getImageDatas(keyword):

    url = 'https://pixabay.com/ko/images/search/'
    cur_dir = os.path.abspath("../images/background/" + keyword + "/")
    main_page_src = requests.get(url + keyword + '/?pagi=1')
    plain_main_page_text = main_page_src.text
    soup_main_page = BeautifulSoup(plain_main_page_text, 'lxml')
    max_page_str = soup_main_page.find('form', {'class': 'add_search_params pure-form hide-xs hide-sm hide-md'})
    for s in max_page_str.get_text().split():
        if s.isdigit():
            max_page = int(s)

    for i in range(1, max_page + 1):
        cur_url = url + keyword + '/?pagi=' + str(i)
        page_src = requests.get(cur_url)
        plain_page_text = page_src.text
        soup_page = BeautifulSoup(plain_page_text, 'lxml')
        for img in soup_page.find_all('img'):
            if "https://cdn.pixabay.com/photo/" in img.get('src'):
                response = requests.get(img.get('src')).content
                file_name = img.get('src')[47:]
                if not os.path.exists(cur_dir):
                    os.makedirs(cur_dir)
                img_file = open(cur_dir + '\\' + file_name, 'wb')
                img_file.write(response)
                img_file.close()



if __name__ == '__main__':

    getImageDatas("도로")