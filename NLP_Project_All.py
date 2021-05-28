#!/usr/bin/env python
# coding: utf-8

# In[299]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer


# In[300]:


data = pd.read_csv('Shoes_NLP.csv',header=None)


# In[301]:


data.rename(columns={0:'brand',1:'stars',2:'review'},inplace=True)
print(data.columns)


# In[302]:


data['review']=data['review'].astype('string')
data['stars']=data['stars'].astype('int')
data['brand']=data['brand'].astype('string')


# In[303]:


data.head(10)


# In[304]:


data['review'].dtypes


# In[305]:


data['review_tok']=data['review'].astype('string')


# In[306]:


alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

data['review_tok'] = data.review.map(alphanumeric).map(punc_lower)
data.head()


# In[307]:


data['review_tok'] =  data['review_tok'].apply(word_tokenize)


# In[308]:


wn = WordNetLemmatizer()
blocker_words = ["shoe","shoes","foot","feet"]
def lemmatization(token_text):
    text = [wn.lemmatize(word) for word in token_text if word not in stop_words and wn.lemmatize(word) not in blocker_words]
    return text


# In[309]:


data['rev_lem'] = data['review_tok'].apply(lambda x : lemmatization(x))


# In[310]:


data.head()


# In[311]:


from sklearn.decomposition import NMF
from sklearn import decomposition
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# In[312]:


stop_words = stopwords.words('english')
stop_words.extend(["shoe", "shoes"])
data['rev_lem']=data['rev_lem'].astype('string')
cv = CountVectorizer(ngram_range=(1,2), binary=True,stop_words='english')
X = cv.fit_transform(data.rev_lem)
pd.DataFrame(X.toarray(), columns=cv.get_feature_names()).head(10)


# In[315]:


nmf_model = NMF(10)
doc_topic = nmf_model.fit_transform(X)


# In[316]:



topic_word = pd.DataFrame(nmf_model.components_.round(3),index = ["component_1","component_2","component_3",
                        "component_4","component_5","component_6","component_7","component_8","component_9",
                        "component_10"],columns = cv.get_feature_names())
       
             
            
topic_word


# In[317]:


def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# In[318]:


display_topics(nmf_model, cv.get_feature_names(), 50)


# In[319]:


nmf_feature_names = cv.get_feature_names()
nmf_weights = nmf_model.components_

print(nmf_model.components_)


# In[320]:


review_topic_matrix = nmf_model.transform(X)


# In[321]:


review_topic_matrix_df = pd.DataFrame(review_topic_matrix).add_prefix('topic_')

review_topic_matrix_df[['raw_reviews', 'clean_reviews']] = data[['review', 'rev_lem']]
review_topic_matrix_df.head()


# In[322]:


for review in review_topic_matrix_df.sort_values(by='topic_0', ascending=False).head(50)['raw_reviews'].values:
    print(review)
    print()


# In[323]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[324]:


get_ipython().system('pip install wordcloud')


# In[325]:


from wordcloud import WordCloud

df = pd.read_csv("wordcloud.csv")
df.head()


# In[326]:


text = " ".join(cat.split()[0] for cat in df.category)
word_cloud = WordCloud(collocations = False, relative_scaling = 0,background_color = 'white',stopwords=["normally","usually", 'got'],).generate(text)


# In[327]:


#Checking for NaN values
#df.isna().sum()
##Removing NaN Values
#df.dropna(inplace = True)
#Creating the text variable

# Creating word_cloud with text as argument in .generate() method

# Display the generated Word Cloud
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
word_cloud.to_file("wordcloud.png")


# In[ ]:




