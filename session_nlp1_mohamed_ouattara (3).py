#!/usr/bin/env python
# coding: utf-8

# In[317]:


#Ouattara_Mohamed_Wenceslas
#Code saisi sur jupiternotebook
#Etape 1
#Definition du repertoire de travail
#Extraction des conclusions
import os
repertoire=os.chdir("C:/Users/USER/Desktop/Session/NLP")


# In[318]:


Conclusion_BEA2016_0759=open("Conclusion_BEA2016_0759.txt").read()
print(Conclusion_BEA2016_0759)


# In[319]:


Conclusion_n_ca001215_05=open("Conclusion_n_ca001215_05.txt").read()
print(Conclusion_n_ca001215_05)
type(Conclusion_n_ca001215_05)


# In[320]:


Conclusion_n_tg000906_06=open("Conclusion_n_tg000906_06.txt").read()
print(Conclusion_n_tg000906_06)


# In[321]:


#Etape2 :Normalisation des donnees/Text normalization
Conclusion_BEA2016_0759_norm=Conclusion_BEA2016_0759.lower()
print(Conclusion_BEA2016_0759_norm)
print('\n')
Conclusion_n_ca001215_05_norm=Conclusion_n_ca001215_05.lower()
print(Conclusion_n_ca001215_05_norm)
print('\n')
Conclusion_n_tg000906_06_norm=Conclusion_n_tg000906_06.lower()
print(Conclusion_n_tg000906_06_norm)


# In[322]:


#Etape 3
#Preparation/nettoyage des donnees/ data cleaning

#Suppression des signes ponctuations
#Suppression des nombres
#Suppression des stopwords
#tonkenization/revoie une liste
import string
import re

from nltk.corpus import stopwords
fr_stopwords=stopwords.words('french')

def data_cleaner(text):
    result1= "".join([word for word in text if word not in string.punctuation])
    result="".join(c for c in result1 if not c.isdigit()) 
    tokens=re.split('\s+',result)
    text=[word for word in tokens if word not in fr_stopwords]
    return text


# In[323]:


Conclusion_BEA2016_0759_clean=data_cleaner(Conclusion_BEA2016_0759_norm)
print(Conclusion_BEA2016_0759_clean)
print('\n')
Conclusion_n_ca001215_05_clean=data_cleaner(Conclusion_n_ca001215_05_norm)
print(Conclusion_n_ca001215_05_clean)
print('\n')
Conclusion_n_tg000906_06_clean=data_cleaner(Conclusion_n_tg000906_06_norm)
print(Conclusion_n_tg000906_06_clean)


# In[324]:


#Stemming
#application du l'algo de stemming sur Conclusion_BEA2016_0759_clean
import nltk
ps=nltk.PorterStemmer()

def conclusion_stemmer(text):
    result=[ps.stem(word) for word in text]
    return result

stem_Conclusion_BEA2016_0759=conclusion_stemmer(Conclusion_BEA2016_0759_clean)
print(stem_Conclusion_BEA2016_0759)


# In[325]:


#application du l'algo de stemming sur Conclusion_n_ca001215_05_clean

import nltk
ps=nltk.PorterStemmer()

stem_Conclusion_n_ca001215_05=conclusion_stemmer(Conclusion_n_ca001215_05_clean)
print(stem_Conclusion_n_ca001215_05)


# In[326]:


#application du l'algo de stemming sur Conclusion_n_tg000906_06_clean

import nltk
ps=nltk.PorterStemmer()

stem_Conclusion_n_tg000906_06=conclusion_stemmer(Conclusion_n_tg000906_06_clean)
print(stem_Conclusion_n_tg000906_06)


# In[327]:


#Etape4_1
#Exploration des donnees/Data exploration
#Word cloud visualization
#application de la Word cloud visualization sur stem_Conclusion_BEA2016_0759
from wordcloud import WordCloud
import matplotlib.pyplot as plt
unique_string=(" ").join(stem_Conclusion_BEA2016_0759)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.close()


# In[328]:


#application de la Word cloud visualization sur stem_Conclusion_n_ca001215_05

unique_string=(" ").join(stem_Conclusion_n_ca001215_05)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.close()


# In[329]:


#application de la Word cloud visualization sur stem_Conclusion_n_tg000906_06

unique_string=(" ").join(stem_Conclusion_n_tg000906_06)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.close()


# In[364]:


import gensim
from gensim import corpora

tokens = [token.split() for token in stem_Conclusion_BEA2016_0759]
gensim_dictionary = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")


# In[365]:


from gensim.models import LdaModel
lda1_h=LdaModel(corpus=corpus_bow,id2word=gensim_dictionary,num_topics=5)
corpus_bow=[gensim_dictionary.doc2bow(doc) for doc in tokens]


# In[366]:


lda1_h.print_topics()


# In[367]:


lda1_h.show_topic(0)


# In[369]:


tokens2 = [token.split() for token in stem_Conclusion_n_ca001215_05]
gensim_dictionary = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")


# In[370]:



lda2_h=LdaModel(corpus=corpus_bow,id2word=gensim_dictionary,num_topics=5)
corpus_bow=[gensim_dictionary.doc2bow(doc) for doc in tokens2]


# In[371]:


lda2_h.show_topic(0)


# In[373]:


#lda2_h.show_topic(0)


# In[374]:


lda2_h.print_topics()


# In[375]:


tokens3 = [token.split() for token in stem_Conclusion_n_tg000906_06]
gensim_dictionary = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")


# In[376]:


lda3_h=LdaModel(corpus=corpus_bow,id2word=gensim_dictionary,num_topics=5)
corpus_bow=[gensim_dictionary.doc2bow(doc) for doc in tokens3]


# In[377]:


lda3_h.print_topics()


# In[379]:


lda3_h.show_topic(0)


# In[380]:


#Etape 4_2
#Application de la vectorisation avec TF-IDF
#Select signifiant words
#application au cas stem_Conclusion_BEA2016_0759
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect1=TfidfVectorizer(analyzer=data_cleaner)
tfidf_vectfinal_Conclusion_BEA2016_0759=tfidf_vect1.fit_transform(stem_Conclusion_BEA2016_0759)

    


# In[381]:


#application au cas stem_Conclusion_BEA2016_0759
tfidf_vect2=TfidfVectorizer(analyzer=data_cleaner)
tfidf_vectfinal_Conclusion_n_ca001215_05=tfidf_vect2.fit_transform(stem_Conclusion_n_ca001215_05)


# In[382]:


#application au cas stem_Conclusion_BEA2016_0759
tfidf_vect3=TfidfVectorizer(analyzer=data_cleaner)
tfidf_vectfinal_Conclusion_n_tg000906_06=tfidf_vect3.fit_transform(stem_Conclusion_n_tg000906_06)


# In[383]:


#transformation en dataframe pour une meilleure lisibilite
#application au cas Conclusion_BEA2016_0759
import pandas as pd
df_tfidf_vectfinal_Conclusion_BEA2016_0759=pd.DataFrame(tfidf_vectfinal_Conclusion_BEA2016_0759.toarray())
print(df_tfidf_vectfinal_Conclusion_BEA2016_0759)


# In[384]:


#application au cas Conclusion_n_ca001215_05

df_tfidf_vectfinal_Conclusion_n_ca001215_05=pd.DataFrame(tfidf_vectfinal_Conclusion_n_ca001215_05.toarray())
print(df_tfidf_vectfinal_Conclusion_n_ca001215_05)


# In[385]:


#application au cas Conclusion_n_tg000906_06
df_tfidf_vectfinal_Conclusion_n_tg000906_06=pd.DataFrame(tfidf_vectfinal_Conclusion_n_tg000906_06.toarray())
print(df_tfidf_vectfinal_Conclusion_n_tg000906_06)


# In[386]:


print(tfidf_vect1.get_feature_names())


# In[387]:


print(tfidf_vect2.get_feature_names())


# In[388]:


print(tfidf_vect3.get_feature_names())


# In[389]:


#application de la Word cloud visualization sur Conclusion_BEA2016_0759 apres TF-IDF

from wordcloud import WordCloud
import matplotlib.pyplot as plt
unique_string=(" ").join(tfidf_vect1.get_feature_names())
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.close()


# In[390]:


#application de la Word cloud visualization sur Conclusion_n_ca001215_05 apres TF-IDF

unique_string=(" ").join(tfidf_vect2.get_feature_names())
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.close()


# In[391]:


#application de la Word cloud visualization sur Conclusion_n_tg000906_06 apres TF-IDF
unique_string=(" ").join(tfidf_vect2.get_feature_names())
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.close()


# In[392]:


#Etape 5
#Topic Modeling avec gensim
#LDA
import gensim
from gensim import corpora
tfidf_vect_list=tfidf_vect1.get_feature_names()
tokens = [token.split() for token in tfidf_vect_list]
gensim_dictionary = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")


# In[393]:



lda1_nlp=LdaModel(corpus=corpus_bow,id2word=gensim_dictionary,num_topics=5)
corpus_bow=[gensim_dictionary.doc2bow(doc) for doc in tokens]


# In[394]:


lda1_nlp.show_topic(0)


# In[395]:


lda1_nlp.print_topics()


# In[396]:


tfidf_vect_list=tfidf_vect2.get_feature_names()
tokens = [token.split() for token in tfidf_vect_list]
gensim_dictionary = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")


# In[397]:


lda2_nlp=LdaModel(corpus=corpus_bow,id2word=gensim_dictionary,num_topics=5)
corpus_bow=[gensim_dictionary.doc2bow(doc) for doc in tokens]


# In[398]:


lda2_nlp.show_topic(0)


# In[399]:


lda2_nlp.print_topics()


# In[400]:


tfidf_vect_list=tfidf_vect3.get_feature_names()
tokens = [token.split() for token in tfidf_vect_list]
gensim_dictionary = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(gensim_dictionary)) + " tokens")


# In[401]:


lda3_nlp=LdaModel(corpus=corpus_bow,id2word=gensim_dictionary,num_topics=5)
corpus_bow=[gensim_dictionary.doc2bow(doc) for doc in tokens]


# In[402]:


lda3_nlp.show_topic(0)


# In[403]:


lda3.print_topics()


# #Conclusion
# #Les conlusions issues du traitement par NLP sont senbiblement les memes que ceux traduits par la realite(langage humains)
# 

# In[ ]:




