#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 15:15:06 2018

@author: kalyan
"""

import pml53
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import cm as cm
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn import datasets

## access the dataset
#Reading data from heart1.csv
heart1 = pd.read_csv('heart1.csv')
heart1=heart1.values
#iris = datasets.load_iris()
X=heart1[:, 0:13]
y=heart1[:,13]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
## Support Vector Machine
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10 ,random_state=1, n_jobs=2)
forest.fit(X_train_std,y_train)

print("Random Forest")
print('Number in test ',len(y_test))
y_pred = forest.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))

y_combined_pred = forest.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())

from sklearn.metrics import accuracy_score
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
#plot_decision_regions(X=X_combined_std, y=y_combined, classifier=forest  , test_idx=range(105,150))
#plt.xlabel('petal length [standardized]')
#plt.ylabel('petal width [standardized]')
#plt.legend(loc='upper left')
#plt.show()