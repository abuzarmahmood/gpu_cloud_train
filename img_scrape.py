"""
Scrape images from wikiart
##############################
1)Iterate through genres
2)Save images in separate folders

https://github.com/robbiebarrat/art-DCGAN/blob/master/genre-scraper.py
"""
########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################
from bs4 import BeautifulSoup 
import requests
import re
import urllib
from tqdm import tqdm,trange
from joblib import Parallel, delayed, cpu_count
import os
import random
import time
from numpy import array_split

genre_pg_url = 'https://www.wikiart.org/en/paintings-by-genre'
html_page = requests.get(genre_pg_url)
soup = BeautifulSoup(html_page.content, 'html.parser')
genre_container = soup.findAll('li', class_="dottedItem")
genre_urls = [x.a['href'] for x in genre_container]
genre_names = [re.findall('e/.+\?',x)[0][2:-1] \
        if "?" in x else re.findall('e/.+',x)[0][2:] for x in genre_urls]

def url_gen(genre,page):
    return f'https://www.wikiart.org/en/paintings-by-genre/{genre}/{page}'

all_urls = []

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def get_url_list(genre_name):
    url_set = set()
    #page_num = 1
    for page_num in trange(1,100):
        # Hacky way to avoid concurrence of requests
        time.sleep(random.random()*3)
        this_url = url_gen(genre_name,page_num)
        #soup = BeautifulSoup(urllib.request.urlopen(this_url), "lxml")
        html_page = requests.get(this_url)
        soup = BeautifulSoup(html_page.content, 'html.parser')
        regex = r'https?://uploads[0-9]+[^/\s]+/\S+\.jpg'
        url_list = re.findall(regex, str(soup.html()))
        dup_count = 0
        for name in url_list:
            if name not in url_set:
                url_set.add(name)
            else:
                dup_count += 1
        if dup_count == len(url_list):
            break
    return genre_name, url_set

all_dat_list = parallelize(get_url_list, genre_names)

all_url_list = [(y[0],x) for y in all_dat_list for x in y[1]]
url_split_list = array_split(all_url_list, cpu_count()-2)
# To maximize parallelization over cores, split list into equal parts

#all_dat_list = [get_url_list(x) for x in genre_names]
#
#abs_url = 'https://www.wikiart.org/en/paintings-by-genre/abstract/100'
save_path = '/home/abuzarmahmood/Desktop/img_conv_net/data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def download_img(img_url, genre, save_path):
    fin_save_path = os.path.join(save_path, genre)
    if not os.path.exists(fin_save_path):
        os.makedirs(fin_save_path)
    global num_downloaded, num_images
    filename = img_url.split('/')[-1]
    #savepath = '%s/%s/%d_%s' % (output_dir, genre, item, filepath[-1])
    try:
        time.sleep(0.2)  # try not to get a 403
        urllib.request.urlretrieve(img_url, 
                os.path.join(fin_save_path, filename))
        #num_downloaded += 1
        #if num_downloaded % 100 == 0:
        #    print('downloaded number %d / %d...' % (num_downloaded, num_images))
    except Exception as e:
        pass
        #print("failed downloading " + str(file), e) 

def download_list(iter_list):
    for this_iter in tqdm(iter_list):
        download_img(this_iter[1], this_iter[0], save_path)

parallelize(download_list, url_split_list)
#html_page = requests.get(abs_url)
#soup = BeautifulSoup(html_page.content, 'html.parser')
#genre_container = soup.findAll('img')
#
#
#pattern = 'https://uploads.+jpg'
#url_list = re.findall(pattern, str(soup.html()))
#
#soup = BeautifulSoup(urllib.request.urlopen(abs_url), "lxml")
#regex = r'https?://uploads[0-9]+[^/\s]+/\S+\.jpg'
#url_list = re.findall(regex, str(soup.html()))
