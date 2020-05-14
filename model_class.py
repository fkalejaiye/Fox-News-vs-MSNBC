import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.metrics import confusion_matrix
import string

class ModelBuilder():
    
    def __init__(self,data):
        self.name ="Model Builder"
        self.articles_info = data
        self.articles = data['content']
        self.porter_stemmer = PorterStemmer()
        
    def stemmer(self,text):
        text = "".join(l for l in text if l not in string.punctuation)
        words = text.lower().split()
        words = [self.porter_stemmer.stem(word) for word in words]
        return words
    
    
    def vectorize(self,num_max_features,ngrams):
        porter_stemmer = PorterStemmer()
        stopwords = nltk.corpus.stopwords.words('english')
        date_corrections = ['breitbart','follow','facebook','twitter','email','coronavirus','tuesday','biden', 'positive','pandemic', '2020','covid','outbreak','health','virus']
        for word in date_corrections:
            stopwords.append(self.porter_stemmer.stem(word))
                       
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,stop_words=stopwords,ngram_range=ngrams,tokenizer=self.stemmer, max_features=num_max_features)
        tfidf = self.vectorizer.fit_transform(self.articles)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(tfidf, self.articles_info['source'],test_size=0.2)
        
    def fit_and_score(self,model_used):
        self.model = model_used
        self.model.fit(self.X_train,self.y_train)
        return self.model.score(self.X_test,self.y_test)
    
    def test_bias(self,text):
    #vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english',max_features=500)
        X = self.vectorizer.transform(text)
        yhat = self.model.predict(X)
        count=0
        for pred in yhat:
            if pred=='breitbart':
                count+=1
        breit_freq = (count/31)*100
        return f'Predicted as Breitbart {round(breit_freq,2)}% of the time.'
    
    def get_most_important_words(self,num_of_words):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_names = self.vectorizer.get_feature_names()
        top_words = []

        for i in range(num_of_words):
            top_words.append(feature_names[indices[i]])
        return top_words
    
    def get_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        return f'True Negatives: {tn}, True Positives {tp}, False Negatives {fn}, False Positives {fp}.'
    
    def find_word_freq_by_class(self,word):
        b=0
        o=0
        for idx,art in enumerate(self.articles_info['content']):
            if word in art:
                if self.articles_info['source'][idx]=='breitbart':
                    b+=1

                elif self.articles_info['source'][idx]=='occupy_democrats':
                    o+=1
        return f'Breitbart has this word {b} times. Occupy Democrats has this word {o} times.'

        
        
        