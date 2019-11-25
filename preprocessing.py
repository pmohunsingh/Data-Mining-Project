import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler

# state column names 
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'the label']
test_raw_nan = pd.read_csv('census-income.test.csv', names = col_names)
test_raw_nan.replace(' ?', np.nan, inplace = True)
# print(test_raw_nan)

#drop missing values 
# test_raw = test_raw_nan.dropna()
# print(test_raw)

# CONVERTING CATEGORICAL VARIABLES TO DUMMY VARIABLES 
test_raw_nan_workclass = pd.get_dummies(test_raw_nan['workclass'])
test_raw_nan_education = pd.get_dummies(test_raw_nan['education'])
test_raw_nan_marital_status = pd.get_dummies(test_raw_nan['marital_status'])
test_raw_nan_occupation = pd.get_dummies(test_raw_nan['occupation'])
test_raw_nan_relationship = pd.get_dummies(test_raw_nan['relationship'])
test_raw_nan_race = pd.get_dummies(test_raw_nan['race'])
test_raw_nan_sex = pd.get_dummies(test_raw_nan['sex'])
test_raw_nan_native_country = pd.get_dummies(test_raw_nan['native_country'])
test_raw_nan_the_label = pd.get_dummies(test_raw_nan['the label'])
# concatenating 
test_raw_nan_concat = pd.concat(
    [test_raw_nan, 
    test_raw_nan_workclass, 
    test_raw_nan_education, 
    test_raw_nan_marital_status, 
    test_raw_nan_occupation,
    test_raw_nan_relationship,
    test_raw_nan_race, 
    test_raw_nan_sex,  
    test_raw_nan_native_country,
    test_raw_nan_the_label],
    axis=1)
print (test_raw_nan_concat.head())

# not sure if i should remove ' Male' and ' ' >=50k' dummy variables 
# dropping categorical columns  
test_raw_nan_concat.drop(['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'the label', 'native_country'], inplace=True, axis=1)
print (test_raw_nan_concat.head())
print(test_raw_nan_concat.columns.values)
