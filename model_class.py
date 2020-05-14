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

class ModelBuilder():
    
    def __init__(self,data):
        self.name ="Model Builder"
        self.articles_info = data
        self.articles = data['content']
        
        
    def vectorize(self,num_max_features):
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.append('breitbart')
        stopwords.append('follow')
        stopwords.append('facebook')
        stopwords.append('twitter')
        stopwords.append('email')
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,stop_words=stopwords,max_features=num_max_features)
        tfidf = self.vectorizer.fit_transform(self.articles)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(tfidf, self.articles_info['source'],test_size=0.2)
        
    def fit_and_score(self,model_used):
        self.model = model_used
        self.model.fit(self.X_train,self.y_train)
        return self.model.score(self.X_test,self.y_test)
    
    def test_bias(self,text):
    #vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english',max_features=500)
        X = self.vectorizer.fit_transform(text)
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
        
        
        
        