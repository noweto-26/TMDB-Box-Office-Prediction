#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Libraries

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))
import os
import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import json
import ast
from urllib.request import urlopen
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


# In[2]:


df = pd.read_csv("final_data_api.csv")


# In[3]:


temp_df = df.copy()


# In[4]:


temp_df.head()


# In[5]:


temp_df.shape


# In[6]:


temp_df.isna().sum() / temp_df.shape[0] >0.3


# In[7]:


temp_df = temp_df.drop(columns='Unnamed: 0', axis=1)


# In[8]:


temp_df = temp_df.drop(columns=['success', 'status_code', 'status_message', 'archive_footage', 'archive_sound', 'imdb_id'], axis=1)


# In[9]:


un_important = ['cinematographer', 'composer', 'production_designer', 'self', 'writer', 'homepage', 'editor']


# In[10]:


temp_df = temp_df.drop(columns=un_important, axis=1)


# In[11]:


temp_df = temp_df.drop(columns=['primaryTitle','id','original_title','tagline','title','overview', 'video', 'revenue', 'vote_count', 'adult', 'startYear','backdrop_path','numVotes','genres_y','poster_path','production_countries','spoken_languages','status'], axis=1)


# In[12]:


temp_df = temp_df.drop(columns=['originalTitle','runtimeMinutes'], axis=1)


# In[13]:


temp_df.isna().sum() / temp_df.shape[0] > 0.3


# In[489]:


# belong to collection
#temp_df = temp_df.drop(['belongs_to_collection'], axis=1)


# In[14]:


temp_df = temp_df.drop(['MPAA'], axis=1)


# In[ ]:





# In[ ]:





# In[491]:


temp_df['belongs_to_collection'].isna().sum() / temp_df.shape[0] # 90 % from data is missing


# In[492]:


#temp_df['belongs_to_collection'] = temp_df['belongs_to_collection'].apply(ast.literal_eval)# .. can't apply it with na value


# In[494]:


#work_df = work_dff.dropna()


# In[495]:


work_df


# 
# # Belong To Collection

# In[496]:


work_df['belongs_to_collection']=work_df['belongs_to_collection'].fillna('{}')


# In[497]:


work_df['belongs_to_collection'].isna().sum()


# In[498]:


#work_df = work_df.dropna()


# In[499]:


work_df['belongs_to_collection'] = work_df['belongs_to_collection'].apply(ast.literal_eval)


# In[500]:


work_df


# In[501]:


#temp_dff = temp_df.dropna().reset_index(drop=True)


# In[502]:


work_df.shape


# In[503]:


# tempdf_ = temp_df.dropna().reset_index(drop=True) .. if you drop na now


# In[504]:


for i, e in enumerate(work_df['belongs_to_collection'][:1077]):
    print(i,e)


# In[505]:


#work_df['belongs_to_collection'][1077]['name']


# In[506]:


list_of_collections = list(work_df['belongs_to_collection'].apply(lambda x : x['name'] if x!={} else []))


# In[507]:


work_df['collection_name'] = work_df['belongs_to_collection'].apply(lambda x: x['name'] if x != {} else 0)
work_df['has_collection'] = work_df['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)


# In[508]:


work_df = work_df.drop(columns=['belongs_to_collection'], axis=1)


# In[509]:


work_df


# # Geners

# In[510]:


for i, e in enumerate(work_df['genres_x'][:5]):
    print(i,e)


# In[511]:


list_of_genres = list(work_df['genres_x'].apply(lambda x : x.split(',')if x !={} else []).values)
list_of_genres


# In[512]:


# Let's  plot World Cloud .. 
plt.figure(figsize = (12, 8))
text = ' '.join([i for j in list_of_genres for i in j])
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top genres Is')
plt.axis("off")
plt.show()


# In[513]:


Counter([i for j in list_of_genres for i in j]).most_common() # orderd by Most Common


# In[514]:



work_df['num_genres'] = work_df['genres_x'].apply(lambda x: len(x.split(",")) if x != {} else 0)
work_df['all_genres'] = work_df['genres_x'].apply(lambda x: ' '.join(sorted(x.split(','))) if x != {} else '')


# In[515]:


work_df


# In[516]:


top_10 = Counter([i for j in list_of_genres for i in j]).most_common(10)
top_10


# In[517]:


top_genres = [i[0] for i in top_10]
top_genres


# In[518]:


for g in top_genres:
    work_df['genre_'+g] = work_df['all_genres'].apply(lambda x : 1 if g in x else 0)


# In[519]:


work_df = work_df.drop(['genres_x'], axis=1)


# In[520]:


work_df


# # Production companies

# In[521]:


work_df['production_companies'] .isna().sum()


# In[522]:


work_df['production_companies'] = work_df['production_companies'].apply(ast.literal_eval) # extract list from Quote


# In[523]:


for i, e in enumerate(work_df['production_companies'][:5]):
    print(i, e)


# In[524]:


x = work_df['production_companies']
x


# In[525]:


#x[0]['name']


# In[526]:


#work_df['production_companies'][0][0]['name']


# In[527]:


list_of_companies = list(work_df['production_companies'].apply(lambda x : [i['name'] for i in x] if x != {} else []).values)
list_of_companies


# In[528]:


# dummy Column .. Contain only name of company
work_df['all_production_companies'] = work_df['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])))


# In[529]:


work_df


# In[530]:


# Let's create column for top 30 production company

#temp_dff['num_companies'] = temp_dff['production_companies'].apply(lambda x: len(x) if x != {} else 0)

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(10)] #20
for g in top_companies:
    work_df['production_company_' + g] = work_df['all_production_companies'].apply(lambda x: 1 if g in x else 0)
    

work_df = work_df.drop(['production_companies', 'all_production_companies'], axis=1)


# In[531]:


work_df


# # Actor, Actress, Director, Producer

# In[532]:


work_df['actor'].isna().sum()


# In[533]:


work_df['actress'].isna().sum()


# In[534]:


work_df['producer'].isna().sum()


# In[535]:


work_df['director'].isna().sum()


# In[536]:


work_df['actor'] = work_df['actor'].fillna('{}')
work_df['actress'] = work_df['actress'].fillna('{}')
work_df['director'] = work_df['director'].fillna('{}')
work_df['producer'] = work_df['producer'].fillna('{}')


# In[537]:


work_df


# In[538]:


for i, e in enumerate(work_df['actor'][:5]):
    print(i, e)


# In[539]:


list_of_actor_names = list(work_df['actor'].apply(lambda x: x.split(',') if x != {} else []).values)
list_of_actor_names


# In[540]:


Counter([i for j in list_of_actor_names for i in j]).most_common(20)


# In[541]:


#work_df['num_cast'] = temp_dff['actor'].apply(lambda x: len(x) if x != {} else 0)
top_cast_names = [m[0] for m in Counter([i for j in list_of_actor_names for i in j]).most_common(20)] #50 # m[0] to return number only
#top_cast_names.remove('{}') 


# In[542]:


top_cast_names


# In[543]:


for g in top_cast_names:
    work_df['actor_name_' + g] = work_df['actor'].apply(lambda x: 1 if g in str(x) else 0)


# In[544]:


work_df = work_df.drop(['actor'], axis=1)


# In[545]:


work_df


# In[546]:


list_of_actress_names = list(work_df['actress'].apply(lambda x: x.split(',') if x != {} else []).values)


# In[547]:


top_cast_names = [m[0] for m in Counter([i for j in list_of_actress_names for i in j]).most_common(20)] #50
top_cast_names.remove('{}')


# In[548]:


for g in top_cast_names:
    work_df['cast_name_' + g] = work_df['actress'].apply(lambda x: 1 if g in str(x) else 0)


# In[549]:


work_df = work_df.drop(['actress'], axis=1)


# In[550]:


list_of_producer_names = list(work_df['producer'].apply(lambda x: x.split(',') if x != {} else []).values)
top_cast_names = [m[0] for m in Counter([i for j in list_of_producer_names for i in j]).most_common(20)] #50
top_cast_names.remove('{}')
for g in top_cast_names:
    work_df['producer_name_' + g] = work_df['producer'].apply(lambda x: 1 if g in str(x) else 0)


# In[551]:


work_df = work_df.drop(['producer'], axis=1)


# In[552]:


list_of_director_names = list(work_df['director'].apply(lambda x: x.split(',') if x != {} else []).values)
top_cast_names = [m[0] for m in Counter([i for j in list_of_director_names for i in j]).most_common(20)] #50
top_cast_names.remove('{}')
for g in top_cast_names:
    work_df['director_name_' + g] = work_df['director'].apply(lambda x: 1 if g in str(x) else 0)


# In[553]:


work_df = work_df.drop(['director'], axis=1)


# In[554]:


#temp_dff=temp_dff.dropna(subset=['release_date'])


# # Revenue

# In[555]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(work_df['world_revenue']);
plt.title('Distribution of revenue');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(work_df['world_revenue']));
plt.title('Distribution of log of revenue');


# In[556]:


# Try to make it close as possible to normal distribution
work_df['log_revenue'] = np.log1p(work_df['world_revenue'])


# # Budget

# In[557]:


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(work_df['budget']);
plt.title('Distribution of Budget');
plt.subplot(1, 2, 2)
plt.hist(np.log1p(work_df['budget']));
plt.title('Distribution of log of Budget');


# In[ ]:





# # Release Date

# In[558]:


work_df['release_date'].isna().sum()


# In[559]:


# Drop rows with nan date
work_df=work_df.dropna(subset=['release_date'])


# In[560]:


work_df['release_date'] = pd.to_datetime(work_df['release_date'])


# In[561]:


def process_date(df):
    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + "_" + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int) # corresponding to pandas date time
    
    return df

work_df = process_date(work_df)


# In[562]:


work_df


# In[563]:


work_df = work_df.drop(['release_date'], axis=1)


# # Original language

# In[564]:


work_df['original_language'].isna().sum()


# In[565]:


l = list(work_df['original_language'].fillna(''))


# In[566]:


le = LabelEncoder()
le.fit(l) 
work_df['original_language'] = le.transform(work_df['original_language'].fillna('').astype(str))


# # collection name

# In[567]:


work_df


# In[568]:


l = list(work_df['collection_name'].fillna(''))


# In[569]:


le = LabelEncoder()
le.fit(l) 
work_df['collection_name'] = le.transform(work_df['collection_name'].fillna('').astype(str))


# In[570]:


# Just drop all geners
work_df = work_df.drop(['all_genres'], axis=1)


# In[571]:


work_df


# #  Eda Section

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # More of suggestion about budget
# 

# In[572]:


work_df[work_df['budget']==0]


# In[573]:


null_budget_columns = work_df[work_df.budget == 0]
null_budget_columns.drop(columns='budget', axis=1, inplace=True)


# In[574]:


null_budget_columns


# In[575]:


train_budget_columns = work_df[work_df.budget != 0]


# In[576]:


train_budget_columns.shape


# In[577]:


train_budget_columns.isna().sum()


# In[578]:


X = train_budget_columns.drop("budget", axis=1).values
y = train_budget_columns.budget.values


# In[579]:


X[0]


# In[580]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[581]:


y[0]


# In[582]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# In[583]:


from xgboost import XGBRFRegressor
model = XGBRFRegressor(objective = "reg:linear", max_deepth=15, seed=100, n_estimators=100, bosster = "gblinear")


# In[584]:


model.fit(X_train, y_train)


# In[585]:


model.score(X_train, y_train)


# In[586]:


model.score(X_test, y_test)


# In[587]:


predictions = model.predict(null_budget_columns.values)


# In[588]:


null_budget_columns = work_df[work_df.budget == 0]


# In[589]:


null_budget_columns.budget = predictions


# In[590]:


null_budget_columns[:5]


# In[591]:


train_budget_columns[:5]


# In[592]:


final_work_df = pd.concat([train_budget_columns, null_budget_columns], axis=0)


# In[593]:


final_work_df


# In[594]:


#final_work_df = final_work_df.drop(columns=['log_budget'],axis = 1)  # Running .... but not now


# In[595]:


final_work_df = final_work_df.drop(columns=['log_revenue'],axis = 1) 


# In[596]:


final_work_df


# In[ ]:





# In[597]:


#mean_value=df['budget'].mean()
#work_df[work_df['budget']==0] = mean_value

#temp_dff = temp_dff.drop(temp_dff[temp_dff['budget'] == 0].index)


# In[598]:


X = final_work_df.drop(['world_revenue'], axis=1)
y = final_work_df['world_revenue']


# In[599]:


from scipy.sparse import csr_matrix

X_ = csr_matrix(X.values)


# In[600]:


X_train, X_valid, y_train, y_valid = train_test_split(X_, y, test_size=0.1)


# In[601]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1,random_state=51 , bootstrap = True,
 max_depth= None,
 max_features = 'auto',
 min_samples_leaf = 7,
 min_samples_split= 5,
 n_estimators = 1000)


# In[602]:


model.fit(X_train, y_train)


# In[603]:


model.score(X_train,y_train)


# In[604]:


model.score(X_valid,y_valid)


# In[605]:


from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(model.predict(X_train),y_train)**0.5


# In[606]:


mean_squared_log_error(model.predict(X_valid),y_valid)**0.5


# In[607]:


# Regressor
#model2 = XGBRFRegressor(objective = "reg:linear", max_deepth=15, seed=100, n_estimators=100, bosster = "gblinear")


# In[608]:


#model2.fit(X_train, y_train)


# In[609]:


#model2.score(X_train,y_train)


# In[610]:


#model2.score(X_valid,y_valid)


# In[611]:


# normal Random forest


# In[ ]:





# In[ ]:





# In[ ]:





# In[612]:


for name, importance in zip(X, model.feature_importances_):
    print(name, "=", importance)


# In[ ]:




