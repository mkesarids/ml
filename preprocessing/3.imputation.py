# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:29:11 2018

@author: mohan
"""

import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import CategoricalImputer

path = 'E:\\MLPractise\\Datasets'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

titanic_train.head(20)
titanic_train.describe()
titanic_train.isnull()

#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer(strategy="mean")
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])