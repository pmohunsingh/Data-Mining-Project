#!/usr/bin/env python
# coding: utf-8

# In[150]:


import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
# state column names 
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'the label']


# In[151]:


test_raw_nan = pd.read_csv('census-income.test.csv', names = col_names)


# In[152]:


#replace the question marks in file with null values, making sure there is a space before the question mark to determine the rows 
test_raw_nan.replace(' ?', np.nan, inplace = True)


# In[153]:


#drop missing values 
test_raw = test_raw_nan.dropna()


# In[154]:


#For the mssing values 
print('test data contains {} rows with missing values'.format(len(test_raw_nan) - len(test_raw)))


# In[155]:


# get_dummies to convert categorical variables into dummy variables
#iloc to retrieve rows from the dataframe
target_nans = pd.get_dummies(test_raw_nan).iloc[:,-1]
target_nans.value_counts()


# In[156]:


print('nagative instances accounted for {}% of our times'.format(1221/(12435+3846)* 100))


# In[ ]:





# In[157]:


test_raw


# In[158]:


#creates a target column from the first column to the one before the last column 
test_target = pd.get_dummies(test_raw).iloc[:, -1]


# In[159]:


#lists the target columns for the dummy variables 
test_target.head()


# In[160]:


#returns the count of the unbalanced data
test_target.value_counts()


# In[161]:


#continous variables collected from first to last column of dataframe
raw_features = test_raw.iloc[:, :-1]


# In[162]:


#the total number of column of raw features 
len(raw_features.columns)


# In[163]:


# concatenates continous variables along one axis
continous = pd.concat([raw_features.age,
                    raw_features.fnlwgt, 
                      raw_features.capital_gain,
                      raw_features.capital_loss,
                      raw_features.hours_per_week], axis=1)


# In[164]:


raw_features = test_raw.iloc[:, :-1]


# In[165]:


#display
raw_features.head()


# In[166]:


workclass = pd.get_dummies(raw_features.workclass)
workclass.head()
len(workclass.columns)


# In[167]:


#checks if workclass has any null values 
workclass.isnull().any()


# In[168]:


#checks if education has any null values 
education = pd.get_dummies(raw_features.education)
education.head()
education.isnull().any()


# In[169]:


#checks if marital has any null values 
marital_status = pd.get_dummies(raw_features.marital_status)
marital_status.head()
marital_status.isnull().any()


# In[170]:


#checks if occupation has any null values 
occupation = pd.get_dummies(raw_features.occupation)
occupation.head()
occupation.isnull().any()


# In[171]:


#checks if relationship has any null values 
relationship = pd.get_dummies(raw_features.relationship)
relationship.head()
relationship.isnull().any()


# In[172]:


#checks if relationship has any null values 
race = pd.get_dummies(raw_features.race)
race.head()
race.isnull().any()


# In[173]:


#checks if sex has any null values 
sex = pd.get_dummies(raw_features.sex)
sex.head()
sex.isnull().any()


# In[174]:


#checks if native_country has any null values 
native_country = pd.get_dummies(raw_features.native_country)
print(native_country)


# In[ ]:





# In[175]:


#add a country because we have been including the second to last column
missing_country = pd.Series(np.zeros(len(native_country)).astype(int))
native_country.insert(loc = 26, column = 'Nigeria', value = missing_country)
native_country.iloc[:,26] = int(0)
len(native_country.columns)


# In[176]:


df1= continous.merge(sex, left_index = True, right_index = True)
df2= df1.merge(race, left_index = True, right_index = True)
df3= df2.merge(relationship, left_index = True, right_index = True)
df4= df3.merge(marital_status, left_index = True, right_index = True)
df5= df4.merge(native_country, left_index = True, right_index = True)
df6= df5.merge(workclass, left_index = True, right_index = True)
df= df6.merge(occupation, left_index = True, right_index = True)
len(df.columns)


# In[177]:


df.insert(loc=88, column = '>50k', value =test_target)


# In[178]:


len(df.columns)


# In[179]:


df.columns = [x.strip() for x in df.columns]


# In[180]:


df.head()


# In[181]:


df.columns


# In[183]:


scaler = StandardScaler()
continous = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
raw_features[continous] = scaler.fit_transform(test_raw[continous])


# In[ ]:




