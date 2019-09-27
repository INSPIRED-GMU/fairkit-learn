import numpy as np
import sklearn as skl
import six
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from fair_metrics import causal_discrimination_score, group_discrimination_score, false_positive_rate_equality, false_negative_rate_equality
from fair_model_selection import FairSearch

from datasets import load_adult_income

import os

os.chdir("fklearn/")


data = load_adult_income()
models = {'LogisticRegression': LogisticRegression}
metrics = {'Causal': group_discrimination_score, 'Accuracy': accuracy_score}
parameters = {
            #   'SVC': {'kernel': ['rbf'], 'C': [1, 10], 'probability': [True]},
              'LogisticRegression': {'penalty': ['l1', 'l2'], 'C': [1, 10]}
              } 

thresholds = [i * 1.0/100 for i in range(10)]
Search = FairSearch(models, metrics, metrics, parameters, thresholds)
Search.fit(data[0])

print(Search)

