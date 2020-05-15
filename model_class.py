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
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF

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
    
    def tokenize(self,text):
        tokenizer = RegexpTokenizer(r"[\w']+")
        stem = PorterStemmer().stem 
        stop_set = set(stopwords.words('english'))
        date_corrections = ['breitbart','follow','facebook','twitter','email','coronavirus','tuesday','biden', 'positive','pandemic', '2020','covid','outbreak','health','virus','joe']
        for word in date_corrections:
            stop_set.add(word)
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token.lower() not in stop_set]
        return stems
    
    def vectorize(self,nummax_features=None,ngrams=None):
        
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize, ngram_range = ngrams,max_features=nummax_features)
        self.tfidf = self.vectorizer.fit_transform(self.articles)
        self.tfidf_breitbart = self.vectorizer.fit_transform(self.articles_info['content'][self.articles_info['source']=='breitbart'])
        self.tfidf_occupy_democrats = self.vectorizer.fit_transform(self.articles_info['content'][self.articles_info['source']=='occupy_democrats'])
        self.vocab = np.array(self.vectorizer.get_feature_names())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tfidf, self.articles_info['source'],test_size=0.2)
        
    
    def fit_nmf_and_return_topics(self,website,num_topics=50,handlabel=False):
        nmf=NMF(n_components=50)
        if website=='breitbart':
            nmf.fit(self.tfidf_breitbart)
            W = nmf.transform(self.tfidf_breitbart)
            H = nmf.components_
            if handlabel==True:
                return hand_label_topics(H,self.vocab,num_topics)
            elif handlabel==False:
                return self.get_topics(H,self.vocab,num_topics)
        elif website=='occupy_democrats':
            nmf.fit(self.tfidf_occupy_democrats)
            W = nmf.transform(self.tfidf_occupy_democrats)
            H = nmf.components_
            if handlabel==True:
                return hand_label_topics(H,self.vocab,num_topics)
            elif handlabel==False:
                return self.get_topics(H,self.vocab,num_topics)
            
        
        
    def get_topics(self, H, vocabulary,num_of_topics):
        count=0
        l = []
        for i, row in enumerate(H):
            if count < num_of_topics:
                top_ten = np.argsort(row)[::-1][:10]
                print(f'Topic {i+1}:')
                print('------->', ' '.join(vocabulary[top_ten]))
                label = f'Topic {i+1}'
                l.append(label)

                count+=1

        return l
    
    
    def fit_and_score(self,model_used):
        self.model = model_used
        self.model.fit(self.X_train,self.y_train)
        return self.model.score(self.X_test,self.y_test)
    
    
    def get_most_important_words(self,num_of_words):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_names = self.vectorizer.get_feature_names()
        top_words = []

        for i in range(num_of_words):
            top_words.append(feature_names[indices[i]])
        return top_words
    
    def get_confusion_matrix(self):
        self.y_pred = self.model.predict(self.X_test)
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        return f'True Negatives: {self.tn}, True Positives {self.tp}, False Negatives {self.fn}, False Positives {self.fp}.'
    
    def find_word_freq_by_class(self,word):
        b=0
        o=0
        for idx,art in enumerate(self.articles_info['content']):
            if word in art or word.capitalize() in art:
                if self.articles_info['source'][idx]=='breitbart':
                    b+=1

                elif self.articles_info['source'][idx]=='occupy_democrats':
                    o+=1
        return f'Breitbart has this word {b} times. Occupy Democrats has this word {o} times.'
    
    def plot_confusion_matrix(self):
        fig,ax = plt.subplots(figsize=(16,8))
        self.cf_matrix = confusion_matrix(self.y_test, self.y_pred)   
        group_names = ['True Neg','False Pos','False Neg','True Pos']   
        group_counts = ["{0:0.0f}".format(value) for value in self.cf_matrix.flatten()]   
        group_percentages = ["{0:.2%}".format(value) for value in self.cf_matrix.flatten()/np.sum(self.cf_matrix)]    
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]   
        labels = np.asarray(labels).reshape(2,2)   
        sns.heatmap(self.cf_matrix,annot=labels, fmt='', cmap='Blues',xticklabels=['Predicted Breitbart', 'Predicted Occ-Dem'],yticklabels=['Actual Breitbart', 'Actual Occ-Dem'])
        plt.yticks(rotation=0) 
        precision = self.tp / (self.tp + self.fp)    
        recall = self.tp / (self.tp + self.fn)    
        accuracy = (self.tp + self.tn) / (self.tn + self.fp + self.fn + self.tp)    
        print(f"Precision: {precision} \nRecal: {recall} \nAccuracy: {accuracy}")

        
        
        