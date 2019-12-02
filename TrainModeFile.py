#!/usr/bin/env python
# coding: utf-8

# In[258]:


import pandas as pd
import numpy as np


# In[259]:


col_names = ['age', 'workclass', 'fnlwgt', 
             'education', 'education_num',
             'marital_status', 'occupation', 
             'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 
             'hours_per_week', 'native_country', '50k']


# In[260]:


df_train_raw = pd.read_csv('census-income.data.csv', names = col_names)


# In[261]:


df_train_raw = df_train_raw.replace(' ?', np.nan)


# In[262]:


df_train_raw.head()


# In[263]:


train_target = pd.get_dummies(df_train_raw).iloc[:,-1]


# In[264]:


train_target.head()


# In[265]:


train_target.value_counts()


# In[266]:


train_features_raw = df_train_raw.iloc[:,:-1]


# In[267]:


len(train_features_raw.columns)


# In[268]:


df_continuous = pd.concat([train_features_raw.age,
           train_features_raw.fnlwgt,
           train_features_raw.capital_gain,
           train_features_raw.capital_loss,
           train_features_raw.hours_per_week], axis=1)


# In[269]:


train_features_raw = df_train_raw.iloc[:,:-1]


# In[270]:


train_features_raw.head()


# In[271]:


train_features_raw.native_country.value_counts()


# In[272]:


train_features_raw.native_country.replace(np.nan, ' United-States', inplace = True)


# In[273]:


df_continuous = pd.concat([train_features_raw.age,
                         train_features_raw.fnlwgt,
                         train_features_raw.capital_gain,
                         train_features_raw.capital_loss,
                         train_features_raw.hours_per_week], axis = 1)


# In[274]:


native_country = pd.get_dummies(train_features_raw.native_country)
native_country.head()


# In[275]:


len(native_country.columns)


# In[307]:


train_features_raw.workclass.value_counts()


# In[308]:


train_features_raw.workclass.replace(np.nan, ' Private', inplace = True)


# In[309]:


workclass = pd.get_dummies(train_features_raw.workclass) 
workclass.head()


# In[310]:


len(workclass.columns)


# In[311]:


train_features_raw.occupation.value_counts()


# In[312]:


train_features_raw.occupation.replace(np.nan, ' Prof-specialty', inplace = True)


# In[313]:


occupation = pd.get_dummies(train_features_raw.occupation) 
len(occupation.columns)


# In[314]:


occupation.head()


# In[315]:


marital_status = pd.get_dummies(train_features_raw.marital_status)
marital_status.head()
len(marital_status.columns)


# In[316]:


relationship = pd.get_dummies(train_features_raw.relationship)
relationship.head()
len(relationship.columns)


# In[317]:


race = pd.get_dummies(train_features_raw.iloc[:,8])
race.head()
len(race.columns)


# In[318]:


sex = pd.get_dummies(train_features_raw.iloc[:,9])
sex.head()


# In[319]:


native_country.columns.get_loc(' Holand-Netherlands')


# In[320]:


df1 = df_continuous.merge(sex, left_index = True, right_index = True)
df2 = df1.merge(race, left_index = True, right_index = True)
df3 = df2.merge(relationship, left_index = True, right_index = True)
df4 = df3.merge(marital_status, left_index = True, right_index = True)
df5 = df4.merge(native_country, left_index = True, right_index = True)
df6 = df5.merge(workclass, left_index = True, right_index = True)
df_mode = df6.merge(occupation, left_index = True, right_index = True)


# In[326]:


len(df_mode.columns)


# In[327]:


df_mode.insert(loc=88, column = '>50k', value =train_target)


# In[328]:


df_mode.columns = [x.strip() for x in df_mode.columns]


# In[329]:


len(df_mode.columns)


# In[330]:


df_mode.head()


# In[331]:


df_mode.columns


# In[332]:


print(df_mode.age.mean(), df_mode.age.std()) 


# In[333]:


df_mode.age = (df_mode.age - df_mode.age.mean())/df_mode.age.std()


# In[334]:


print(df_mode.fnlwgt.mean(), df_mode.fnlwgt.std())


# In[335]:


df_mode.fnlwgt = (df_mode.fnlwgt - df_mode.fnlwgt.mean())/df_mode.fnlwgt.std()


# In[336]:


print(df_mode.capital_gain.mean(), df_mode.capital_gain.std())


# In[337]:


df_mode.capital_gain = (df_mode.capital_gain - df_mode.capital_gain.mean()) / df_mode.capital_gain.std()


# In[338]:


print(df_mode.capital_loss.mean(), df_mode.capital_loss.std())


# In[339]:


df_mode.capital_gain = (df_mode.capital_loss  - df_mode.capital_loss.mean())/df_mode.capital_loss.std()


# In[340]:


print(df_mode.hours_per_week.mean(), df_mode.hours_per_week.std())


# In[341]:


df_mode.hours_per_week = (df_mode.hours_per_week - df_mode.hours_per_week.mean()) / df_mode.hours_per_week.std()


# In[342]:


df_mode.head()


# In[344]:


df_mode.to_csv('train_mode_file.csv')


# In[ ]:


df.

