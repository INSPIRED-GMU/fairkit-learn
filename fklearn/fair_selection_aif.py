"""
Used to search and return models along the Pareto frontier using AIF360 metrics
"""
import six
import itertools
import numpy as np
import tensorflow as tf

from fklearn.ml_pipeline import MLPipeline
from fklearn.fair_model_selection import filtered_arguments
from aif360.algorithms.inprocessing import AdversarialDebiasing

# MANDATORY hyperparmaeters for adversarial debiasing
def DEFAULT_ADB_PARAMS(privileged, unprivileged):
    """
    Create a dictionary of mandatory hyperparameters for adversarial debiasing
    """
    
    return {'unprivileged_groups': [unprivileged], 'privileged_groups': [privileged],
            'scope_name': ['adb'], 'sess': [tf.Session()]}

class ModelSearch(object):
    """

    Parameters
    ----------
    models : dict
        Dictionary of model names as keys and instantiations of model objects as values.
        e.g. {
              'SVC': sklearn.svm.SVC(), 
              'LogisticRegression': sklearn.linear_model.LogisticRegression()
                } 

    metrics : dict[str, (MetricClass, str)]
        Dictionary of sklearn/AIF360 fairness metrics. The keys are the display names of the metrics, and
        the values are 2-tuples with the first element containing the metric class object, and the second
        containing the name of the metric function to evaluate.
        e.g. {
               'ClassificationMetric': (aif360.metrics.ClassificationMetric, 'num_generalized_true_positives'), 
               'BinaryLabelDatasetMetric': (aif360.metrics.BinaryLabelDatasetMetric, 'disparate_impact')
             }

    hyperparameters : dict of dicts of lists
        Dictionary with model names as keys and hyperparameter dicts as values.
        Each hyperparameter dict has hyperparameters as keys and hyperparameter 
        settings to try as values.
        e.g. {
              'SVC': {'kernel': ['rbf'], 'C': [1, 10]},
              'LogisticRegression': {'penalty': ['l1', 'l2'], 'C': [1, 10]}
              } 
    

    thresholds : list of floats
        List of classifation thresholds to be applied to all classifiers.
        Usage is for classifiers that output a probability, rather than a
        hard classification.
        e.g. [i * 1.0/100 for i in range(100)]
    """

    def __init__(self, models, metrics, hyperparameters, thresholds):
        self.models = models
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.thresholds = thresholds
        self.search_results = []
        self.pareto_optimal_results = []
    
    def grid_search(self, dataset, privileged=[], unprivileged=[], test_frac=0.3, preprocessors=[], postprocessors=[]):
        """
        Performs a grid search over the specified model + hyperparameter pairs, calculating metric evalutations for each model.

        Args:
            dataset (aif360.datasets.StructuredDataset): An instance of a structured dataset
            test_frac (float): A real number between 0 and 1 denoting the % of the dataset to be used as test data
            privileged (list[dict]): A list of dictionaries containing privileged groups
            unprivileged (list[dict]): A list of dictionaries containing unprivileged groups
        """
        
        # If any pre/postprocessors were supplied, add the option for None by default
        preprocessors += [None]
        postprocessors += [None]

        self.model_id = 0
        
        # Try each unique model
        for model_name, ModelClass in six.iteritems(self.models):

            # If no hyperparameters were specified, use the defaults. Otherwise setup a grid search
            if len(self.hyperparameters[model_name]) == 0:
                param_list = [{}]
            else:
                parameter_keys, parameter_values = zip(*self.hyperparameters[model_name].items())
                param_list = [dict(zip(parameter_keys, v)) for v in itertools.product(*parameter_values)]
            
            # Grid search through hyperparameters in the current model
            for param_set in param_list:

                model = ModelClass(**param_set)

                # Go through each combination of pre/post processing algorithms
                for preprocessor, postprocessor in itertools.product(preprocessors, postprocessors):
       
                    mlp = MLPipeline(model, privileged=privileged, unprivileged=unprivileged, preprocessor=preprocessor, postprocessor=postprocessor)
                    
                    # Create a new search result for each threshold value
                    for threshold in self.thresholds:

                        if model_name == 'AdversarialDebiasing':
                            mlp.model.scope_name = str(self.model_id)
                            self.model_id += 1

                            mlp.model.sess.close()
                            tf.reset_default_graph()
                            mlp.model.sess = tf.Session()

                        mlp.fit(dataset, test_frac=test_frac, threshold=threshold)
                        search_result = {'model_class': model_name,
                                        'hyperparameters': param_set,
                                        'preprocessor': type(preprocessor).__name__ if preprocessor else 'None',
                                        'postprocessor': type(postprocessor).__name__ if postprocessor else 'None',
                                        'metrics': {}
                                        }
                        
                        # Populate metrics for this search result
                        for metric_name, metric_args in six.iteritems(self.metrics):

                            # The first metric argument is the Metric Class itself. The rest are the names of
                            # submetric evaluation functions
                            MetricClass = metric_args[0]

                            for metric_fn in metric_args[1:]:
                                metric_val = mlp.evaluate(MetricClass, metric_fn)
                                metric_category = '{} ({})'.format(metric_name, metric_fn)
                                search_result['metrics'][metric_category] = metric_val

                        self.search_results.append(search_result)

        self.pareto_optimal_results = self.__filter_solution_set()

    def __filter_solution_set(self):
        # Inspired by https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        assert(self.search_results)

        costs = -1 * np.array([[v for _, v in six.iteritems(result['metrics'])] for result in self.search_results])

        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points

        return [result for i, result in enumerate(self.search_results) if is_efficient[i]]

    def to_csv(self, filename):
        """
        Exports the search results as a CSV file
        
        Args:
            filename (str): The name of the file to save the results to
        Raises:
            AttributeError: If a grid search has not yet been performed, an AttributeError will be raised
        """

        import csv

        if len(self.search_results) == 0:
            raise AttributeError("A grid search must be performed before exporting results to CSV")
        
        # Compute CSV headers for all metrics in the search results
        metric_headers = { metric for res in self.pareto_optimal_results for metric in res['metrics'] }
     
        with open(filename, mode='w') as csv_file:
            headers = ['model', 'hyperparameters', 'preprocessor', 'postprocessor', *list(metric_headers)]
            writer = csv.DictWriter(csv_file, fieldnames=headers, lineterminator='\n')
            writer.writeheader()

            for result in self.pareto_optimal_results:
                metric_dict = {metric_name: metric_val for metric_name, metric_val in six.iteritems(result['metrics'])}
                
                writer.writerow({'model': result['model_class'],
                                 'preprocessor': result['preprocessor'],
                                 'postprocessor': result['postprocessor'],
                                 'hyperparameters': repr(result['hyperparameters'] or 'Default (see sklearn docs)'),
                                 **metric_dict})

        
