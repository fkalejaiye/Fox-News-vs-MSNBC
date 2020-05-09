import numpy as np
from pymongo import MongoClient
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time


pages = np.arange(1,401)

client = MongoClient()
breitbart_db=client['breitbart']
urls = breitbart_db['urls']
articles = breitbart_db['articles']

def get_urls(x):
    links=[]
    count=0
    first_half = "https://www.breitbart.com"
    for num in x:
        pg = requests.get(f'https://www.breitbart.com/politics/page/{num}/')
        if pg.status_code==200:
            soup = BeautifulSoup(pg.text,"lxml")
            data = soup.findAll('div', {'class':"tC"})
        else:
            continue
            
        for i in data:
            link=first_half+(i.find('a').get("href"))
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
            title = soup.find('h1').text
            author = soup.find('div', {"class": 'header_byline'}).find('a').text
            date = soup.find('div', {"class": 'header_byline'}).find('time').text
            shares = soup.find('span', {"class": 'acz5'}).text
            
            data = soup.findAll('div', {'class':"entry-content"})
            content=[]
            for i in data:
                for j in i.find_all('p'):
                    t = j.get_text()
                    content.append(t)
            content = " ".join(content)
            key = {"title":title}
            value = {'title':title, 'author':author, 'date': date, 'content': content}
            articles.update(key,value,upsert=True)
            count+=1
            print(f'Scraped {count} articles. ({title})')
            time.sleep(2)
        else:
            print('Failed to get page!')
            continue

websites = [urls.find()[i]['link'] for i in range(5000,7500)]
get_articles(websites)
print('Finished Scraping!')