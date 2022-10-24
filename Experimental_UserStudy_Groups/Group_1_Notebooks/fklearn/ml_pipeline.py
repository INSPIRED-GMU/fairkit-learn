import numpy as np
from sklearn.preprocessing import StandardScaler

from aif360.metrics import ClassificationMetric
from aif360.algorithms import Transformer
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

class MLPipeline(object):

    """
    Defines a machine-learning pipeline for evaluating fairness in predictors. For usage, see example at the bottom of the file.

    Args:
        model (sklearn.model | aif360.algorithms.inprocessing): An sklearn predictor OR an AIF360 inprocessing algorithm
        privileged (list[dict[str, float]]): A list of dictionaries with keys representing privileged attribute + value pairs
        unprivileged (list[dict[str, float]]): A list of dictionaries with keys representing unprivileged attribute + value pairs
        preprocessor (aif360.algorithms.preprocessing): An instance of an AIF360 preprocessing algorithm
        postprocessor (aif360.algorithms.postprocessing): An instance of an AIF360 postprocessing algorithm
    """

    def __init__(self, model, privileged=[], unprivileged=[], preprocessor=None, postprocessor=None):
        self.model = model
        self.privileged = privileged
        self.unprivileged = unprivileged
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataset_train = []
        self.dataset_test = []
        self.test_predictions = []


    def fit(self, dataset, test_frac=0.3, threshold=0.5, feature_scaling=False):
        """
        Trains our model on the dataset. Uses different control flow depending on if we are using an
        sklearn model or an AIF360 inprocessing algorithm

        Args:
            dataset (aif360.datasets.StructuredDataset): An instance of a structured dataset
            test_frac (float): A real number between 0 and 1 denoting the % of the dataset to be used as test data
            threshold (float): A real number between 0 and 1 denoting the threshold of acceptable class imbalance
        """

        if test_frac < 0 or test_frac > 1:
            raise ValueError("Parameter test_frac must be between 0 and 1")

        dataset_train, dataset_test = dataset.split([1-test_frac], shuffle=False)

        # If a preprocessing algorithm was supplied, apply that transformations first
        if self.preprocessor:
            dataset_train = self.preprocessor.fit_transform(dataset_train)
            dataset_test = self.preprocessor.fit_transform(dataset_test)

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        
        self.__fit_inprocessing(threshold, feature_scaling)

    def __fit_inprocessing(self, threshold, feature_scaling):
        """
        Trains an AIF360 inprocessing model on the provided dataset.

        Args:
        """
        
        # Apply feature scaling if specified
        if feature_scaling:
            scaler = StandardScaler().fit(self.dataset_train.features)
            self.dataset_train.features = scaler.fit_transform(self.dataset_train.features)
            self.dataset_test.features = scaler.transform(self.dataset_test.features) 

        self.model.fit(self.dataset_train)
   
        
        # Make our predictions, without thresholds for now
        dataset_test_pred = self.model.predict(self.dataset_test)

        # If a postprocessing algorithm was specified, transform the test results
        if self.postprocessor:
            dataset_test_pred = self.postprocessor.fit(self.dataset_test, dataset_test_pred) \
                                                  .predict(dataset_test_pred)

        self.classified_data = dataset_test_pred

    
    def evaluate(self, metric, submetric):
        """
        Evaluates an AIF360 metric against the trained model. 
        
        Args:
            metric (aif360.metrics.Metric): An AIF360 metric class
            submetric (str): A string denoting the metric evaluation function that is to be called on the provided metric class
        Returns:
            float: A float denoting the performance of each method evaluation within the specified metric on the trained model
        Raises:
            AttributeError: If a model has not been trained yet, or
                            If the provided submetric function does not exist on the metric class, or
                            If the provided submetric function contains arguments other than "privileged"

        """
        
        from inspect import signature
        import re
        
        if not self.dataset_train:
            raise AttributeError("A model must be fit before evaluating a metric")

        curr_metric = metric(self.dataset_test, self.classified_data, unprivileged_groups=self.unprivileged, privileged_groups=self.privileged)
        
        # Retrieve the callable evalation function 'submetric' of this metric instance
        submetric_fn = getattr(curr_metric, submetric)
        
        return submetric_fn()
