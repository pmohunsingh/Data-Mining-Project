#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
# state column names 
col_names = ['age', 'workclass', 'fnlwgt',
             'education', 'education_num', 
             'marital_status', 'occupation', 
             'relationship', 'race', 'sex', 
             'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country', 'the label']


# In[32]:


test_raw_nan = pd.read_csv('census-income.test.csv', names = col_names)


# In[33]:


#replace the question marks in file with null values, making sure there is a space before the question mark to determine the rows 
test_raw_nan.replace(' ?', np.nan, inplace = True)


# In[34]:


#drop missing values 
test_raw = test_raw_nan.dropna()


# In[35]:


#For the mssing values 
print('test data contains {} rows with missing values'.format(len(test_raw_nan) - len(test_raw)))


# In[36]:


# get_dummies to convert categorical variables into dummy variables
#iloc to retrieve rows from the dataframe
target_nans = pd.get_dummies(test_raw_nan).iloc[:,-1]
target_nans.value_counts()


# In[37]:


print('nagative instances accounted for {}% of our times'.format(1221/(12435+3846)* 100))


# In[38]:


test_raw


# In[39]:


#creates a target column from the first column to the one before the last column 
test_target = pd.get_dummies(test_raw).iloc[:, -1]


# In[40]:


#lists the target columns for the dummy variables 
test_target.head()


# In[41]:


#returns the count of the unbalanced data
test_target.value_counts()


# In[42]:


#continous variables collected from first to last column of dataframe
raw_features = test_raw.iloc[:, :-1]


# In[43]:


#the total number of column of raw features 
len(raw_features.columns)


# In[44]:


# concatenates continous variables along one axis
continous = pd.concat([raw_features.age,
                    raw_features.fnlwgt, 
                      raw_features.capital_gain,
                      raw_features.capital_loss,
                      raw_features.hours_per_week], axis=1)


# In[45]:


raw_features = test_raw.iloc[:, :-1]


# In[46]:


#display
raw_features.head()


# In[47]:





# In[59]:


#add a country because we have been including the second to last column
missing_country = pd.Series(np.zeros(len(native_country)).astype(int))
native_country.insert(loc = 26, column = 'Holand-Netherlands', value = missing_country)
native_country.iloc[:,26] = int(0)
len(native_country.columns)


# In[60]:


len(native_country.columns)


# In[66]:


raw_features.workclass.value_counts()


# In[67]:


workclass = pd.get_dummies(raw_features.workclass)
workclass.head()


# In[65]:


len(workclass.columns)


# In[69]:


occupation = pd.get_dummies(raw_features.occupation)
len(occupation.columns)


# In[70]:


occupation.head()


# In[71]:


marital_status = pd.get_dummies(raw_features.marital_status)
len(marital_status.columns)


# In[72]:


marital_status.head()


# In[73]:


relationship = pd.get_dummies(raw_features.relationship)
len(relationship.columns)


# In[74]:


relationship.head()


# In[75]:


race = pd.get_dummies(raw_features.race)
len(race.columns)


# In[76]:


race.head()


# In[77]:


sex = pd.get_dummies(raw_features.iloc[:,9])
sex.head()


# In[78]:


native_country.columns.get_loc('Holand-Netherlands')


# In[80]:


df1 = continous.merge(sex, left_index = True, right_index = True)
df2 = df1.merge(race, left_index = True, right_index = True)
df3 = df2.merge(relationship, left_index = True, right_index = True)
df4 = df3.merge(marital_status, left_index = True, right_index = True)
df5 = df4.merge(native_country, left_index = True, right_index = True)
df6 = df5.merge(workclass, left_index = True, right_index = True)
df = df6.merge(occupation, left_index = True, right_index = True)
len(df.columns)


# In[82]:


df.insert(loc = 87, column = '>50k', value = test_target)


# In[83]:


df.columns = [x.strip() for x in df.columns]
len(df.columns)


# In[84]:


df.head()


# In[85]:


def z_normalize(x):
    print(x.mean(), x.std())
    z = (x - x.mean())/x.std()
    return z


# In[86]:


df.age = z_normalize(df.age)


# In[87]:


df.fnlwgt = z_normalize(df.fnlwgt)


# In[88]:


df.capital_gain = z_normalize(df.capital_gain)


# In[89]:


df.capital_loss = z_normalize(df.capital_loss)


# In[91]:


df.hours_per_week = z_normalize(df.hours_per_week)


# In[92]:


df.head()


# In[93]:


df.to_csv('dropna_test.csv')


# In[ ]:




