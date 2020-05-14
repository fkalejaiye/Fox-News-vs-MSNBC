import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import datetime


class DataPipeline():
    
    
    def __init__(self):
        self.name = 'Article Data'
        self.articles = pd.Series()
        self.articles_info = pd.DataFrame()

    def pipeline(self,news_source):
        client = MongoClient()
        db = client[f'{news_source}']
        articles = db['articles']
        df = pd.DataFrame(list(articles.find()))
        df = df.iloc[:,1:]
        df['title']=[title.strip() for title in df['title']]
        df = df.drop_duplicates()
        df['source']=f'{news_source}'
        df.reset_index(inplace=True)
        df = df.iloc[:,1:]
        new_content2=[]
        for art in df['content']:
            new_text = []
            for sentence in art.split("."):
                if "Follow" not in sentence and "is a reporter" not in sentence and "Breitbart" not in sentence and "breitbart" not in sentence and "reporting" not in sentence and "comment section" not in sentence and "what YOU have to say" not in sentence:
                    new_text.append(sentence)
            new_content2.append(".".join(new_text))
        df['content']= new_content2
        return df
    
    def combine_news_sources(self,source1,source2):
        self.articles_info = pd.concat([self.pipeline(source1),self.pipeline(source2)])
        self.articles = self.articles_info['content']
        self.articles_info = self.articles_info.reset_index().iloc[:,1:]


    def get_word_count(self,article):
            return len(article.split())
    
    def plot_word_counts(self):
        word_count_oc=[]
        word_count_breit= []
        for idx,article in enumerate(self.articles):
            if self.articles_info['source'].iloc[idx]=='occupy_democrats':
                word_count_oc.append(self.get_word_count(article))
            else:
                word_count_breit.append(self.get_word_count(article))

        fig,ax = plt.subplots(1,2,figsize=(16,8),sharey=True)
        ax[0].hist(word_count_oc,color="b",label="Occupy Democrats")
        ax[1].hist(word_count_breit,label="Breitbart")
        ax[0].set_xlabel("Word Count")
        ax[0].set_ylabel("Frequency")
        ax[1].set_xlabel("Word Count")
        plt.suptitle("Word Counts for each Website")
        fig.legend();

    def plot_top_ten_authors(self,website,c):
        website_top_ten = self.articles_info[self.articles_info['source']==website].groupby(['author','source']).count().sort_values('date',ascending=False)['title'].iloc[0:10]
        print(website_top_ten)
        fig,ax=plt.subplots(figsize=(16,8))
        website_top_ten.plot(kind='barh',color=c,alpha=0.5)
        ax.set_ylabel("Author")
        ax.set_xlabel("Number of Articles Published")
        plt.title(f'{website.capitalize()} Top Ten Authors')
        plt.tight_layout();


    def plot_article_dates(self,website):
        self.articles_info['date'] = pd.to_datetime(self.articles_info['date'])
        date_counts = self.articles_info[self.articles_info['source']==website].sort_values('date').groupby('date').count()
        fig,ax =plt.subplots(figsize=(16,8))
        plt.hist(date_counts.index,bins=20)
        plt.xlabel("Year")
        plt.ylabel("Frequency")
        plt.title("Article Date Frequency");
    
 