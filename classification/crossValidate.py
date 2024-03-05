# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:13:02 2018

@author: mohan
"""

import pandas as pd
import os
from sklearn import tree, model_selection

path = 'E:\\MLPractise\\Datasets'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())
features = ['Parch','SibSp']
X_train = titanic_train[features]
y_train = titanic_train[['Survived']]
classifer = tree.DecisionTreeClassifier()
classifer.fit(X_train,y_train)
results = model_selection.cross_validate(classifer, X_train, y_train, cv = 10)
print("Test Mean Score :",results.get('test_score').mean())
print("Train Mean Score :",results.get('train_score').mean())


