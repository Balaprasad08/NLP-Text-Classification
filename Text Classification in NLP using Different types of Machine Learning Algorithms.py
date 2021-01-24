#!/usr/bin/env python
# coding: utf-8

# ### Import Important Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('E:\\prasad\\practice\\My Working Projects\\Completed\\NLP\\Text Classification in NLP using ML(SVM)')


# ### Perform Imports and Load Data

# In[3]:


df=pd.read_table('smsspamcollection.tsv',sep='\t')
df.head(2)


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


plt.figure(figsize=(5,3))
sns.heatmap(df.isnull())
plt.show()


# In[7]:


df.info()


# In[8]:


df.describe()


# ### Visualize the data:

# In[9]:


sns.countplot(df.label)
plt.show()


# In[10]:


df.label.value_counts()


# ### Data Split into Train,Test

# In[11]:


X=df['message']
X.head(2)


# In[12]:


y=df['label']
y.head(2)


# In[13]:


from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ### Featuer Extraction of NLP

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.pipeline import Pipeline


# ### Model Building

# In[16]:


svm=Pipeline([('tfidf',TfidfVectorizer()),
                 ('svm',SVC())])
svm.fit(X_train,y_train)


# In[17]:


y_pred=svm.predict(X_test)


# In[18]:


y_pred


# In[19]:


accuracy_score(y_test,y_pred)


# In[20]:


confusion_matrix(y_test,y_pred)


# ### Model Evaluation

# #### Create Function For Model Evaluation

# In[21]:


def check_model(clf,X_train,X_test,y_train,y_test):
    model=Pipeline([('tfidf',TfidfVectorizer()),('clf',clf)])
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('accuracy_score:',accuracy_score(y_test,y_pred))
    print('\n')
    print('CM:',confusion_matrix(y_test,y_pred))
    print('\n')
    print('Classification_Report:',classification_report(y_test,y_pred))


# In[22]:


check_model(SVC(),X_train,X_test,y_train,y_test)


# ### Check Accuracy_score by using different algorithms

# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC


# In[24]:


# LogisticRegression
check_model(LogisticRegression(),X_train,X_test,y_train,y_test)


# In[25]:


# RandomForestClassifier
check_model(RandomForestClassifier(),X_train,X_test,y_train,y_test)


# In[26]:


# KNeighborsClassifier
check_model(KNeighborsClassifier(),X_train,X_test,y_train,y_test)


# In[27]:


# DecisionTreeClassifier
check_model(DecisionTreeClassifier(),X_train,X_test,y_train,y_test)


# In[28]:


# MultinomialNB
check_model(MultinomialNB(),X_train,X_test,y_train,y_test)


# In[29]:


# SVC
check_model(SVC(),X_train,X_test,y_train,y_test)


# #### Support Vector Classifier Predict Best Accuracy-accuracy_score: 0.986244019138756

# In[30]:


y_pred=svm.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[31]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[32]:


sns.heatmap(cm,annot=True)


# In[33]:


emails = ["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
         "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info",
         "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.",
         "Oh k...i'm watching here:)",
         "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/Ãº1.20 POBOXox36504W45WQ 16+",
         "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL"]


# In[34]:


svm.predict(emails)


# ### Save Model in Pickle & Joblib

# In[35]:


import pickle,joblib


# In[36]:


pickle.dump(svm,open('nlp_text_clf_pkl','wb'))


# In[37]:


joblib.dump(svm,'nlp_text_clf_jbl')


# ### Load Pickle Model

# In[38]:


model_pkl=pickle.load(open('nlp_text_clf_pkl','rb'))
y_pred_pkl=model_pkl.predict(X_test)
print(accuracy_score(y_test,y_pred_pkl))


# ### Load Joblib Model

# In[39]:


model_jbl=joblib.load('nlp_text_clf_jbl')
y_pred_jbl=model_jbl.predict(X_test)
print(accuracy_score(y_test,y_pred_jbl))


# In[40]:


confusion_matrix(y_test,y_pred_jbl)


# In[ ]:




