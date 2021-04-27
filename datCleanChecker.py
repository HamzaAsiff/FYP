# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 00:04:37 2021

@author: Ayesha
"""

import pandas as pd
from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk import sent_tokenize
#nltk.download('punkt')
#nltk.download('stopwords')

def inputSeq(novelname):
    # load data
    #filename = 'thevanishinghalf.txt'
    file = open(novelname,  encoding="utf8")
    text = file.read()
    file.close()

    # split into words
    tokens = word_tokenize(text)
    
    # remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]

    #lowercase all words
    words = [word.lower() for word in words]

    stop_words = stopwords.words('english')
    #print(stop_words)

    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    
    return words
    
    '''
    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    #print(stemmed[:100])
    
    #return stemmed[:len(stemmed)]
    return stemmed[:100]
    '''
    
vanishWord = inputSeq('thevanishinghalf.txt')
#vanishSent = outputSeq('thevanishinghalf.txt')
#print(vanishSent)
print(len(vanishWord))
#print(len(vanishSent))


repWord = inputSeq('Republic.txt')
#repSent = outputSeq('Republic.txt')
#print(repSent)
print(len(repWord))
#print(len(repSent))


glassWord = inputSeq('glass.txt')
#glassSent = outputSeq('glass.txt')
#print(glassSent)
print(len(glassWord))
#print(len(glassSent))


evolWord = inputSeq('evol.txt')
#evolSent = outputSeq('evol.txt')
#print(evolSent)
print(len(evolWord))
#print(len(evolSent))


sportWord = inputSeq('sports.txt')
#sportSent = outputSeq('sports.txt')
#print(sportSent)
print(len(sportWord))
#print(len(sportSent))


animalWord = inputSeq('animal.txt')
#animalSent = outputSeq('animal.txt')
#print(animalSent)
print(len(animalWord))
#print(len(animalSent))


totalWords = vanishWord + repWord + glassWord + evolWord + sportWord + animalWord
#totalSentences = vanishSent + repSent + glassSent + evolSent + sportSent + animalSent

print(len(totalWords))

inputData = str(totalWords)

#outputData = str(totalSentences)

file = open("inputSeq.txt", 'wb')
file.write(inputData.encode('utf-8', 'ignore'))
file.close()

f = open('inputSeq.txt', 'r+')
n = f.read().replace(',', ',\n') 
f.truncate(0)                    
f.write(n)                       
f.close()

#Make CSV
read_file = pd.read_csv (r'inputSeq.txt')
read_file.to_csv (r'inputSeq.csv', index=None)

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
    if i in inputData:
        yes += 1
    else:
        no += 1
        #print(i)
    
print(yes)
print(no)

# Load the files for processing
file_1 = open('inputSeq.csv', encoding = "utf-8")
file_2 = open('outputSeq.csv', encoding = "utf-8")

# Prep some empty sets to throw words into
words_1 = set()
words_2 = set()

for word in file_1.read().split():
    cleaned_word = ''.join([i for i in list(word.lower()) 
        if i.isalpha() or i == "'"
    ])
    if cleaned_word != '': # Just in case!
        words_1.add(cleaned_word)

for word in file_2.read().split():
    cleaned_word = ''.join([
        i for i in list(word.lower()) 
        if i.isalpha() or i == "'"
    ])
    if cleaned_word != '': # Just in case!
        words_2.add(cleaned_word)

similar_words = words_1 & words_2
print(len(similar_words))
