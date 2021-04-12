# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:56:11 2021

@author: Ayesha
"""

from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk import sent_tokenize

nltk.download('punkt')
#nltk.download('stopwords')


def dataClean(novelname):
    # load data
    #filename = 'thevanishinghalf.txt'
    file = open(novelname,  encoding="utf8")
    text = file.read()
    file.close()

    # split into sentences

    sentences = sent_tokenize(text)

    print(len(sentences))

    print(sentences[1])
    
    # split into words
    
    tokens = word_tokenize(text)
    print(tokens[:100])

    # remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]
    print(words[:100])

    #lowercase all words
    words = [word.lower() for word in words]
    print(words[:100])


    stop_words = stopwords.words('english')
    print(stop_words)

    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    print(words[:100])
    
    df = pd.read_csv('myCSV.csv')
    #Get Img Labels
    imglblList = []
    for column in df[['Image_Label']]:
        imglbl = df[column]
    for i in imglbl.values:
        imglblList.append(i)
        
    lbl = set(imglblList)
    sorted(lbl)
    
    yes = 0
    no = 0
    
    print(len(lbl))
    
    for i in lbl:
        if i in words:
            yes += 1
        else:
            no += 1
    
    print(yes)
    print(no)
    
    
        # Total Tokens/Words
    print(len(words))
    
    # Unique Tokens/Words
    print(len(set(words)))
    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    #print(stemmed[:100])
    
    #return stemmed[:len(stemmed)]
    return stemmed[:100]
   
    






result = dataClean('thevanishinghalf.txt') + dataClean('Republic.txt')
print(result)

