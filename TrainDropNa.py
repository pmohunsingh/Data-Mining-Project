#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np


# In[63]:


col_names = ['age', 'workclass', 'fnlwgt', 
             'education', 'education_num',
             'marital_status', 'occupation', 
             'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 
             'hours_per_week', 'native_country', '50k']


# In[64]:


train_raw_nans = pd.read_csv('census-income.data.csv', names = col_names)


# In[65]:


train_raw_nans.replace(' ?', np.nan, inplace = True)


# In[66]:


df_train_raw = train_raw_nans.dropna()


# In[67]:


print('training data set has {} rows with missing values'.format(len(train_raw_nans) - len(df_train_raw)))


# In[68]:


target_nans = pd.get_dummies(train_raw_nans).iloc[:,-1]
target_nans.value_counts()


# In[69]:


print('{}% of our instances have missing values'.format(2399/(24720+7481)*100))


# In[70]:


print('negative instances accounted for {}% of our times'.format(24720/(24720+7481)*100))


# In[71]:


print('positive instances accounted for {}% of our times'.format(7481/(24720+7481)*100))


# In[72]:


df_train_raw.head()


# In[73]:


#encodes target columns that have missing values
train_target = pd.get_dummies(df_train_raw).iloc[:,-1]


# In[74]:


train_target.head()


# In[75]:


train_target.value_counts()


# In[76]:


train_features_raw = df_train_raw.iloc[:,:-1]


# In[77]:


len(train_features_raw.columns)


# In[78]:


df_continous = pd.concat([train_features_raw.age,
                         train_features_raw.fnlwgt,
                         train_features_raw.capital_gain,
                         train_features_raw.capital_loss,
                         train_features_raw.hours_per_week], axis = 1)


# In[79]:


train_features_raw = df_train_raw.iloc[:, :-1]


# In[80]:


train_features_raw.head()


# In[81]:


native_country = pd.get_dummies(train_features_raw.native_country)
native_country.head()


# In[82]:


len(native_country.columns)


# In[83]:


train_features_raw.workclass.value_counts()


# In[84]:


workclass = pd.get_dummies(train_features_raw.workclass)
workclass.head()


# In[85]:


len(workclass.columns)


# In[86]:


occupation = pd.get_dummies(train_features_raw.occupation)
occupation.head()


# In[87]:


len(occupation.columns)


# In[88]:


marital_status = pd.get_dummies(train_features_raw.marital_status)
marital_status.head()


# In[89]:


len(marital_status.columns)


# In[90]:


relationship = pd.get_dummies(train_features_raw.relationship)
relationship.head()


# In[91]:


len(relationship.columns)


# In[92]:


race = pd.get_dummies(train_features_raw.race)
race.head()


# In[93]:


sex = pd.get_dummies(train_features_raw.iloc[:,9])
sex.head()


# In[94]:


native_country.columns.get_loc(' Holand-Netherlands')


# In[95]:


df1 = df_continous.merge(sex, left_index = True, right_index = True)
df2 = df1.merge(race, left_index = True, right_index = True)
df3 = df2.merge(relationship, left_index = True, right_index = True)
df4 = df3.merge(marital_status, left_index = True, right_index = True)
df5 = df4.merge(native_country, left_index = True, right_index = True)
df6 = df5.merge(workclass, left_index = True, right_index = True)
df = df6.merge(occupation, left_index = True, right_index = True)


# In[96]:


len(df.columns)


# In[97]:


df.insert(loc = 87, column = '>50k', value = train_target)


# In[98]:


df.columns = [x.strip() for x in df.columns]


# In[99]:


df.head()


# In[100]:


#normalizing the continous 
def z_normalize(x):
    print(x.mean, x.std())
    z = (x - x.mean())/x.std()
    return z


# In[103]:


df.age = (df.age - 38.437901995888865) / 13.134664776856338


# In[104]:


df.fnlwgt = (df.fnlwgt - 189793.83393011073) / 105652.97152851959


# In[105]:


df.capital_gain = (df.capital_gain  - 1092.0078575691268) / 7406.346496681988


# In[106]:


df.capital_loss = (df.capital_loss  - 88.37248856176646) / 404.2983704862744


# In[107]:


df.hours_per_week = (df.hours_per_week - 40.93123798156621) / 11.979984229273281


# In[108]:


df.head()


# In[109]:


df.to_csv('train_dropna.csv')


# In[ ]:




