import numpy as np
import sklearn as skl
import six
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from aif360.datasets import AdultDataset, GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult
from aif360.algorithms.preprocessing import DisparateImpactRemover, LFR, Reweighing, OptimPreproc
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification

from fair_selection_aif import AIF360Search, DEFAULT_ADB_PARAMS

import os

dataset = GermanDataset()
models = {'LogisticRegression': LogisticRegression, 'KNeighborsClassifier': KNeighborsClassifier}
metrics = {'ClassificationMetric': [ClassificationMetric,
               'num_generalized_true_positives',
               'num_true_negatives',
               'false_positive_rate',
               'false_negative_rate',
               'generalized_false_positive_rate'
           ]
        #    'BinaryLabelDatasetMetric': [BinaryLabelDatasetMetric, 'disparate_impact']
          }
unprivileged = [{'age': 0, 'sex': 0}]
privileged = [{'age': 1, 'sex': 1}]
preprocessor_args = {'unprivileged_groups': unprivileged, 'privileged_groups': privileged}

# Hyperparameters may either be specified as a dictionary of string to lists, or by an empty dictionary to
# use the default ones set by sklearn (or AIF360). The keys are the names of the hyperparameters, and the
# values and lists of possible values to form a grid search over
parameters = {
              'LogisticRegression': {'penalty': ['l1', 'l2'], 'C': [0.1, 0.5, 1]},
              'KNeighborsClassifier': {}
             }
thresholds = [i * 10.0/100 for i in range(5)]
preprocessors=[DisparateImpactRemover(), Reweighing(**preprocessor_args)]
postprocessors=[CalibratedEqOddsPostprocessing(**preprocessor_args), EqOddsPostprocessing(**preprocessor_args), RejectOptionClassification(**preprocessor_args)]

Search = AIF360Search(models, metrics, parameters, thresholds)
Search.grid_search(dataset, privileged=privileged, unprivileged=unprivileged, preprocessors=preprocessors, postprocessors=postprocessors)

Search.to_csv("interface/static/data/test-file.csv")

