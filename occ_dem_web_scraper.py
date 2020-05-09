import numpy as np
from pymongo import MongoClient
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

pages = np.arange(1,668)
client = MongoClient()
occupy_democrats_db=client['occupy_democrats']
urls = occupy_democrats_db['urls']
articles = occupy_democrats_db['articles']

def get_urls(x):
    links=[]
    for num in x:
        pg = requests.get(f'https://occupydemocrats.com/category/politics/page/{num}/')
        if pg.status_code==200:
            soup = BeautifulSoup(pg.text,"lxml")
            data = soup.findAll('div', {'class':"post-title"})
        else:
            print(pg.status_code)
            continue
            
        for i in data:
            link=(i.find('a').get("href"))
            if link not in links:
                links.append(link)          
        print(f'Scraped {len(links)} urls.')
        time.sleep(2)
    for link in links:
        urls.insert_one({'link': link})
    print("Done!!")

def get_articles(urls_of_articles):
    count=0
    for url in urls_of_articles:
        pg = requests.get(url)
        if pg.status_code==200:
            soup = BeautifulSoup(pg.text,"lxml")
            title = soup.find('h1').text.strip()
            author = soup.find('div', {"class": 'author-content'}).find('a').text
            date = soup.find('div', {"class": 'thb-post-date'}).text.strip()
            data = soup.findAll('div', {'class':"post-content entry-content"})
            content=[]
            for i in data:
                for j in i.find_all('p'):
                    t = j.get_text()
                    content.append(t)
                    
            content = " ".join(content)
            articles.insert_one({'title':title, 'author':author, 'date': date, 'content': content})
            count+=1
            print(f'Scraped {count} articles. ({title})')
            time.sleep(2)
            
        else:
            continue

websites = [urls.find()[i]['link'] for i in range(5001,6000)]
get_articles(websites)
print('Finished Scraping!')