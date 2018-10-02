# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:55:52 2018

@author: mohan
"""


import seaborn as sns
import pandas as pd
import os
print(sns.__version__)

path = 'E:\\MLPractise\\Datasets'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

###############################  Univariate Analysis  ###########################

#categorical columns: numerical EDA
pd.crosstab(index=titanic_train["Survived"], columns="count")
pd.crosstab([titanic_train["Survived"],titanic_train["Pclass"]], columns="count")
pd.crosstab(index=titanic_train["Pclass"], columns="count")  
pd.crosstab(index=titanic_train["Sex"],  columns="count")


#continuous features: numerical EDA
titanic_train['Fare'].describe()



#categorical columns: visual EDA
sns.countplot(x='Survived',data=titanic_train)
sns.countplot(x='Pclass',data=titanic_train)
sns.countplot(x='Sex',data=titanic_train)


#continuous features: visual EDA
sns.boxplot(x='Fare',data=titanic_train)
sns.distplot(titanic_train['Fare'])
sns.kdeplot(titanic_train['Fare'])
sns.distplot(titanic_train['Fare'], kde=False)
sns.distplot(titanic_train['Fare'], bins=20, rug=True, kde=False)
sns.distplot(titanic_train['Fare'], bins=100, kde=False)



###############################  Bivariate Analysis  ###########################

#explore bivariate relationships: categorical vs categorical 
sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#explore bivariate relationships: continuous  vs categorical
sns.FacetGrid(titanic_train, hue="Survived",size=6).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, col="Survived",size=6).map(sns.distplot, "Fare").add_legend()
#sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.boxplot, "Age").add_legend()

#explore bivariate relationships: continuous vs continuous 
sns.jointplot(x="Age", y="Fare", data=titanic_train)# -*- coding: utf-8 -*-


###############################  Multivariate Analysis  ###########################

#is age have an impact on survived for each sex group
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex" ).map(sns.kdeplot, "Age")

#is age have an impact on survived for each pclass and sex group?
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived").map(sns.kdeplot, "Age").add_legend()

g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age")

g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(plt.scatter, "Fare", "Age").add_legend()

sns.heatmap(titanic_train[['Fare','Age','Parch']])



