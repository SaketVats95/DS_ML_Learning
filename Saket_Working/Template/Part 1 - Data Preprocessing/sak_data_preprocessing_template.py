# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:44:42 2020

@author: saket.vats
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

#Taking care of missing data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])


#Taking care of categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoding_X=LabelEncoder()
X[:,0]=labelEncoding_X.fit_transform(X[:,0])
onehotEncoder_X = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X=onehotEncoder_X.fit_transform(X)



