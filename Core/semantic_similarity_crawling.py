import configparser
import requests
import matplotlib.pylab as plt
import os
import queue
from bs4 import BeautifulSoup


def test():
    config = configparser.ConfigParser()
    config.read('../config.ini')

    url = 'http://swoogle.umbc.edu/SimService/GetSimilarity'
    idx = 0
    phrase1 = ""
    phrases1 = []
    phrases2 = []
    similarity_scores = queue.Queue()
    for item in config.items('CONSISTENCY'):
        for tags in item:
            if idx % 2 == 1:
                phrase2_tmp = []
                for phrase2 in tags.replace('\n', ' ').split(', '):
                    page_src = requests.post(url,
                                             params={'operation': 'phrase_sim', 'phrase1': phrase1, 'phrase2': phrase2,
                                                     'sim_type': 'relation',
                                                     'corpus': 'webbase',
                                                     'query': 'Get Similarity'})
                    soup_page_src = BeautifulSoup(page_src.text, 'lxml')
                    similarity_score_src = soup_page_src.find('textarea')
                    similarity_score = similarity_score_src.get_text().split('score is ')[1][:-1]
                    similarity_scores.put(similarity_score)
                    phrase2_tmp.append(phrase2)
                phrases2.append(phrase2_tmp)
            else:
                phrase1 = tags
                phrases1.append(tags)
            idx += 1



    csv_columns = ['phrase1', 'phrase2', 'score']
    csv_file = os.path.abspath("../semantic_similarity_score.csv")
    try:
        with open(csv_file, 'w', encoding='utf-8-sig') as csvfile:
            csvfile.write(f"{csv_columns[0]},{csv_columns[1]},{csv_columns[2]}\n")
            idx = -1
            for phrases_2 in phrases2:
                idx += 1
                for phrase2 in phrases_2:
                    csvfile.write(f"{phrases1[idx]},{phrase2},{similarity_scores.get()}\n")
    except IOError:
        print("I/O error")


def create_graph(background_keyword):
    config = configparser.ConfigParser()
    config.read('../config.ini')
    matching_list = config['CONSISTENCY'][background_keyword].replace('\n', ' ').split(', ')

    url = 'http://swoogle.umbc.edu/SimService/GetSimilarity'
    phrases2 = []
    similarity_scores = []
    for phrase2 in matching_list:
        page_src = requests.post(url,
                                 params={'operation': 'phrase_sim', 'phrase1': background_keyword, 'phrase2': phrase2,
                                         'sim_type': 'relation',
                                         'corpus': 'webbase',
                                         'query': 'Get Similarity'})
        soup_page_src = BeautifulSoup(page_src.text, 'lxml')
        similarity_score_src = soup_page_src.find('textarea')
        similarity_score = similarity_score_src.get_text().replace('\n', ' score is ').split('score is ')[1][:-1]
        similarity_scores.append(float(similarity_score))
        phrases2.append(phrase2)

    print(similarity_scores)

    plt.figure(figsize=(19, 10))
    plt.title(f"{background_keyword} and objects")
    plt.plot(phrases2, similarity_scores)
    plt.show()
