# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:02:09 2021

@author: Hank Xiong
"""

from sklearn.datasets import load_iris
import numpy as np
from TreeDef import myDecisionTreeClassifier

iris = load_iris()
X = iris['data']
y = iris['target']

## using this dataset to build my own classifier
MAX_DEPTH = 1
MIN_SAMPLES_LEAF = 4

socre = np.float64('inf')

a = myDecisionTreeClassifier(MAX_DEPTH = 3,metrics_type = 'Gini',MAX_FEATURES= None,MIN_SAMPLES_LEAF=8 )
a.fit(X,y)
a.lhs.unique_classes
a.lhs.probs
tmp = a.predict_probability(X)
a.predict_probability(X[:20,:])
a.predict_probability(X[40:75,:])

a.probs
a.rhs.rhs.classes

for xi in np.nditer(X):
  print(xi)
  
?np.nditer
