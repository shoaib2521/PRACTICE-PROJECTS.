#!/usr/bin/env python
# coding: utf-8

# # RED WINE QUALITY PREDICTION PROJECT

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score

url = "https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Red%20Wine/winequality-red.csv"
wine_data = pd.read_csv(url, sep=';')

print(wine_data.head())
print(wine_data.info())

wine_data = wine_data.iloc[:, 0].str.split(',', expand=True)
wine_data.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol', 'quality']

wine_data = wine_data.apply(pd.to_numeric, errors='coerce')

print(wine_data.head())
print(wine_data.info())

cutoff = 7
wine_data['quality_label'] = np.where(wine_data['quality'] >= cutoff, 1, 0)

print(wine_data['quality_label'].value_counts())

X = wine_data.drop(['quality', 'quality_label'], axis=1)
y = wine_data['quality_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best AUC score on validation set: {:.2f}".format(grid_search.best_score_))

y_pred = grid_search.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}".format(accuracy))
print("Classification Report:")
print(classification_report(y_test, y_pred))

y_prob = grid_search.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score: {:.2f}".format(roc_auc))

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2)
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[ ]:




