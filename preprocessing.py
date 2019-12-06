import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler

# assigning the columns names 
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'the label']
test_raw_nan = pd.read_csv('census-income.test.csv', names = col_names)
# print(test_raw_nan)
test_raw_nan.replace(' ?', np.nan, inplace = True)
# print(test_raw_nan)

# creating the dummy variables
# dropping the first dummy variable of each categorical feaure 
workclass = pd.get_dummies(test_raw_nan['workclass'], drop_first=True)
education = pd.get_dummies(test_raw_nan['education'], drop_first=True)
marital_status = pd.get_dummies(test_raw_nan['marital_status'], drop_first=True)
occupation = pd.get_dummies(test_raw_nan['occupation'], drop_first=True)
relationship = pd.get_dummies(test_raw_nan['relationship'], drop_first=True)
race = pd.get_dummies(test_raw_nan['race'], drop_first=True)
sex = pd.get_dummies(test_raw_nan['sex'], drop_first=True)
native_country = pd.get_dummies(test_raw_nan['native_country']) # one country is already missing, no need to drop first
the_label = pd.get_dummies(test_raw_nan['the label'], drop_first=True)

# concatenating 
test_concat = pd.concat(
    [test_raw_nan, 
    workclass, 
    education, 
    marital_status, 
    occupation,
    relationship,
    race, 
    sex,  
    native_country,
    the_label],
    axis=1)
# print (test_concat)

# dropping the categorical variables as well as education_num
test_concat.drop(['workclass', 'education', 'marital_status', 'education_num', 'occupation', 'relationship', 'race', 'sex', 'the label', 'native_country'], inplace=True, axis=1)
# renaming the Male dummy variable as 'sex' and the >50k dummy variable as 'the label'
test_concat.rename(columns = {' Male':'sex', ' >50K.':'the label'}, inplace = True)
print (test_concat)
# print(test_concat.columns.values)

test_concat.to_csv('train_notdroped.csv')
