#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:29:21 2021

@title: Text Analytics Project
@author: robertseybold

Made with guidance from: https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
"""

#%% Import Libraries
import urllib
import re
from bs4 import BeautifulSoup as soup
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import string
from sklearn.metrics.pairwise import cosine_similarity


#%% Read URl in for the data
url = urllib.request.urlopen('https://www.gutenberg.org/files/2591/2591-h/2591-h.htm#link2H_4_0026')
doc = url.read()

#%% Extracts the separate chapters, gets the text, and adds them to a main list
tree = soup( doc,'lxml' )
#Gets all the div = chapter classes
div = tree.find_all( 'div', { 'class': 'chapter' } )

#Takes the div and gets only the text without the header
text_list = []
header_list = []
for i in range(len(div)):
    #Gets all paragraph tags for a given paragraph
    header_list.append([j.getText().replace('\r',"").replace('\n'," ").strip() for j in div[i].find_all('h2')])
    temp_text_list = div[i].find_all(['p','pre'])

    #Temporarily stores the full story to then be appended to the text list
    story_temp = ""
    #Loops through all paragraphs and concatenate them together
    for k in temp_text_list:
        k = k.getText().replace('\r',"").replace('\n'," ").strip().split()
        k = ' '.join(k)
        story_temp = story_temp + ' ' + k
    #Append the full story to the final text list
    text_list.append(story_temp)

#Removes the blank from the list so this is just the relevant chapters.
del text_list[0]
del header_list[0]

#%% Create the term vectors for each story (we actually don't use this I think)
#Will be used for second deliverable
term_vectors = []

#Populates the term_vectors list with 62 term vectors
#NOTE: these are term vectors for the ENTIRE document
for i in range(len(text_list)):
    #Remove the punctuation and replace the new line characters
    t = re.sub(r'[^\w\s]', '', text_list[i]).replace('\r\n',"").replace('—',' ')
    #Sets the string to lowercase, strip the whitespace, and split at the whitespace
    tok = t.lower().strip().split()


    #Conducts porter stemming on the words
    porter = nltk.stem.porter.PorterStemmer()
    stem = []
    for word in tok:
        stem.append(porter.stem(word))


    #Dictionary of the terms
    d = {}
    for term in stem:
        d[term] = (1 if term not in d else d[term] + 1)

    #Append the full vector for that story to a final list to be used later
    d1 = {}
    print(d1)
    for i in d:
        if i in stopwords.words('english'):
            continue
        else:
            d1[i] = d[i]

    term_vectors.append(d1)

#%%Test of summarization using extractive summation and Cosine distance
 
def read_article(stories):
    #NOTE: The apostrophe is not standard it is : (’)
    #NOTE: A different hyphen is used to separate noise or who abrupt end (—)
    article = re.split(r"\? |\! |\. |\.\’ |\.\’ |\!\”\’ ",stories)
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))

    #deletes the odd black space at the start of each story
    del sentences[0][0]
    #print(sentences)
    return sentences
    

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    #Gets the words in the sentences and sends them to lowercase
    #need to replace all punctuation
    sent1 = [re.sub(r'[^\w\s]', '',w.lower().replace('—',' ').replace('‘',' ').replace('-',' ')) for w in sent1]
    sent2 = [re.sub(r'[^\w\s]', '',w.lower().replace('—',' ').replace('‘',' ').replace('-',' ')) for w in sent2]

    #stemming of the words
    porter = nltk.stem.porter.PorterStemmer()
    stem1 = []
    stem2 = []
    for word in sent1:
        stem1.append(porter.stem(word))

    for word in sent2:
        stem2.append(porter.stem(word))
 
    all_words = list(set(stem1 + stem2))


    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the term vector for the first sentence
    for w in stem1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in stem2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    #print(similarity_matrix[1])
    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)


    #pagerank_numpy is deprecated
    scores = nx.pagerank_numpy(sentence_similarity_graph)


    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)       

    #Full list of all the sentences that are combines
    sentences_full = []
    #Populates the list
    for i in sentences:
        comb_sent = ' '.join(i)
        sentences_full.append(comb_sent)
    

    #Just find the ranked sentences we want to use and adds to list
    ranked_sent_list = []
    for i in range(top_n):
        cur_sent = " ".join(ranked_sentence[i][1])
        ranked_sent_list.append(cur_sent)
    
    #Creates the story by looping through to see if a sentence is in the ranked list
    for sent in sentences_full:
        if sent in ranked_sent_list and sent not in summarize_text:
            summarize_text.append(sent)


    with open('/Users/robertseybold/Desktop/MSA_General/Text_Analytics/Similarity.txt', 'w') as f:
        #f.write(sentence_similarity_martix)
        np.savetxt(f, sentence_similarity_martix, newline='\n')

    # output the summarize text
    #Comment out when outputting to a file 
    print("Summarize Text: \n", ". \n\n".join(summarize_text))


#Calls the survey function
#Gets the summary for just one story. Mainly used for isolating and debugging. Comment out when generating all surveys
generate_summary(text_list[11],5)

#Combines all the summaries together into one list to be output
#Commented out to allow for intdividual output
""" all_sum = []
for m in range(len(text_list)):
    print(m)
    all_sum.append(generate_summary(text_list[m],5))
 """


# %%
#Outputs the name, termvectors, and summary of each story 
#The final output as a text file
""" with open('/Users/robertseybold/Desktop/MSA_General/Text_Analytics/Summaries_of_Fairy_Tales.txt', 'w') as f:
    for i in range(len(header_list)):
        dic2=dict(sorted(term_vectors[i].items(),key= lambda x:x[1],reverse = True))
        f.write(str(header_list[i]))
        f.write('\n\n')
        f.write('Most Frequent Terms:')
        f.write('\n\n')
        f.write(str(dic2))
        f.write('\n\n\n')
        f.write('Summary:')
        f.write('\n\n')
        f.write(all_sum[i])
        f.write('\n\n\n\n\n')

# %%
with open('/Users/robertseybold/Desktop/MSA_General/Text_Analytics/Texts.txt', 'w') as f:
    f.write(str(text_list)) """
# %%
