# Packages for Web-Scraping
import os
import sys

from requests import get
from bs4 import BeautifulSoup
from time import time
from time import sleep
from random import randint
from warnings import warn
import pandas as pd
import csv
# Packages for Saving File after Scraping
import numpy as np
from numba import njit, cuda
import time

def get_last_index(id_):
    url = "https://archiveofourown.org/works/" + str(id_) + "/kudos?before=4577931054"
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    last_index_container = html_soup.find('ol', class_="pagination actions")
    if  last_index_container:
        index_list = last_index_container.find_all('li')
        last_index = index_list[-2].get_text(strip=True)
        return last_index
    else:
        return 0


# get the last page index

kudos_df = []

#@njit(parallel=True)
def get_kudos(id_):
    last_page_index = get_last_index(id_)
    kudos = []
    for i in range(1, int(last_page_index)):
        time.sleep(5)
        working_url = "https://archiveofourown.org/works/" + str(id_) + "/kudos?before=4577931054&page=" + str(i)
        response = get(working_url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        kudos_container = html_soup.find('div', class_="kudos-index region")
        if type(kudos_container).__name__ == 'NoneType':
            continue
        kudos_div = kudos_container.find('p', class_="kudos")
        if type(kudos_div).__name__ != 'NoneType':
            kudos_people = kudos_div.find_all('a')
            for kudos_person in kudos_people:
                person = kudos_person.get_text(strip=True)
                kudos.append(person)
    return kudos

#@njit(parallel=True)
def kudos_loop(ids):
    with open('kudos.csv', 'a', newline="") as f_out:
        writer = csv.writer(f_out)
        with open('errors_kudos.csv', 'a', newline="") as e_out:
            errorwriter = csv.writer(e_out)
            # does the csv already exist? if not, let's write a header row.
            for id in ids:
                result = get_kudos(id)
                try:
                    writer.writerow([id]+[result])
                    print([id]+[result], ' ok')
                except:
                    print('Unexpected error: ', sys.exc_info()[0])
                    error_row = [id] + [sys.exc_info()[0]]
                    errorwriter.writerow(error_row)
                    print(id, 'error')
                time.sleep(5)


df=pd.read_csv('ids_3.csv',header=None)
ids=df.iloc[:,0].values.tolist()
tic = time.time()
kudos_loop(ids)
toc=time.time()
print(toc-tic)