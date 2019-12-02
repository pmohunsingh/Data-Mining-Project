#!/usr/bin/env python
# coding: utf-8

# In[275]:


import pandas as pd
import numpy as np


# In[276]:


col_names = ['age', 'workclass', 'fnlwgt', 
             'education', 'education_num',
             'marital_status', 'occupation', 
             'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 
             'hours_per_week', 'native_country', '50k']


# In[277]:


df_test_raw = pd.read_csv('census-income.test.csv', names = col_names)


# In[278]:


df_test_raw.replace(' ?', np.nan, inplace = True)


# In[279]:


df_test_raw.head()


# In[280]:


#encodes target columns that have missing values
test_target = pd.get_dummies(df_test_raw).iloc[:,-1]


# In[281]:


test_target.head()


# In[282]:


test_target.value_counts()


# In[283]:


test_features_raw = df_test_raw.iloc[:,:-1]


# In[284]:


len(test_features_raw.columns)


# In[285]:


df_continuous = pd.concat([test_features_raw.age,
                         test_features_raw.fnlwgt,
                         test_features_raw.capital_gain,
                         test_features_raw.capital_loss,
                         test_features_raw.hours_per_week], axis=1)


# In[286]:


test_features_raw = df_test_raw.iloc[:, :-1]


# In[287]:


test_features_raw.head()


# In[288]:


test_features_raw.native_country.value_counts()


# In[289]:


test_features_raw.native_country.replace(np.nan, ' United-States', inplace = True)


# In[290]:


native_country = pd.get_dummies(test_features_raw.native_country)
native_country.head()


# In[291]:


missing_country = pd.Series(np.zeros(len(test_features_raw), dtype = int))


# In[292]:


native_country.insert(loc = 14, column = ' Holand-Netherlands', value = missing_country)


# In[293]:


len(native_country.columns)


# In[294]:


test_features_raw.workclass.value_counts()


# In[295]:


test_features_raw.workclass.replace(np.nan, ' Private', inplace = True)


# In[296]:


missing_workclass = pd.Series(np.zeros(len(test_features_raw), dtype = int))


# In[297]:


workclass = pd.get_dummies(test_features_raw.workclass)


# In[298]:


workclass.head()


# In[299]:


len(workclass.columns)


# In[300]:


test_features_raw.occupation.value_counts()


# In[301]:


test_features_raw.occupation.replace(np.nan, ' Prof-specialty', inplace = True)


# In[302]:


occupation = pd.get_dummies(test_features_raw.occupation)
occupation.head()


# In[303]:


len(occupation.columns)


# In[304]:


marital_status = pd.get_dummies(test_features_raw.marital_status)
marital_status.head()

len(marital_status.columns)


# In[305]:


relationship = pd.get_dummies(test_features_raw.relationship)
relationship.head()

len(relationship.columns)


# In[306]:


race = pd.get_dummies(test_features_raw.iloc[:,8])
race.head()

len(race.columns)


# In[307]:


sex = pd.get_dummies(test_features_raw.iloc[:,9])
sex.head()


# In[308]:


len(native_country.columns)


# In[309]:


df1 = df_continuous.merge(sex, left_index = True, right_index = True)
df2 = df1.merge(race, left_index = True, right_index = True)
df3 = df2.merge(relationship, left_index = True, right_index = True)
df4 = df3.merge(marital_status, left_index = True, right_index = True)
df5 = df4.merge(native_country, left_index = True, right_index = True)
df6 = df5.merge(workclass, left_index = True, right_index = True)
df_mode = df6.merge(occupation, left_index = True, right_index = True)


# In[310]:


len(df_mode.columns)


# In[311]:


df_mode.insert(loc = 88, column = '>50k', value = test_target)


# In[312]:


len(df_mode.columns)


# In[313]:


df_mode.columns = [x.strip() for x in df_mode.columns]


# In[314]:


df_mode.head()


# In[315]:


df_mode.age = (df_mode.age - 38.76832669322709) / 13.380675582270195


# In[316]:


df_mode.fnlwgt = (df_mode.fnlwgt - 189616.37025232404) / 105615.00652318467


# In[317]:


df_mode.capital_gain = (df_mode.capital_gain  - 1120.301593625498) / 7703.181842367612


# In[318]:


df_mode.capital_gain = (df_mode.capital_loss  - 89.04189907038513) / 406.28324537681925


# In[319]:


df_mode.hours_per_week = (df_mode.hours_per_week - 40.951593625498006) / 12.062831369168284


# In[320]:


df_mode.head()


# In[321]:


df_mode.to_csv('test_mode_file.csv')


# In[ ]:




