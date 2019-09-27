"""
Used to search and return models along the Pareto frontier

Inspiration: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV 
"""
from __future__ import division
import warnings
import six
import itertools
import inspect
import numpy as np 
import numpy.ma as ma

RANDOM = 'random'
GRID = 'grid'
THRESHOLD_STR = 'threshold'

SEARCH_STRATEGIES = [GRID, RANDOM]

def filtered_arguments(func):
	required_args = six.viewkeys(inspect.signature(func).parameters)
	
	def inner(*args, **kwargs):
		kwargs = { k:v for k,v in six.iteritems(kwargs) if k in required_args }
		return func(*args, **kwargs)
	return inner

class FairSearch():
	"""
	Description 
	TODO 

	Parameters
	----------
	models : dict
		Dictionary of model names as keys and instantiations of model objects as values.
		e.g. {
			  'SVC': sklearn.svm.SVC(), 
			  'LogisticRegression': sklearn.linear_model.LogisticRegression()
				} 

	metrics : dict
	Dictionary of sklearn and fklearn fairness metrics
		e.g. {
			  'Causal': fklearn.fair_metrics.causal_discrimination_score, 
			  'Accuracy': sklearn.metrics.accuracy_score
			}

	parameters : dict of dicts of lists
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

	Attributes 
	----------
	pareto_optimal_results : dict of masked arrays
			Keys strings describing the model parameter or score metric.
			e.g. {'param_C': masked_array(data = [0, --], mask = [False, True]),
				  'param_L1': masked_array(data = [0, --], mask = [False, True]),
				  'train_causal_fairness_score' : [0.8, 0.7],
				  'val_causal_fairness_score' : [0.71, 0.64],
				  'test_causal_fairness_score' : [0.7, 0.65],
				  'train_accuracy_score' : [0.6, 0.8],
				  'val_accuracy_score' : [0.57, 0.81],
				  'test_accuracy_score' : [0.55, 0.78],
				  'fit_time' : [0.08, 1.1]}

	Examples
	--------
	>>> from fklearn.fair_model_selection import FairSearch
	"""

	def __init__(self, models, fairness_metrics, performance_metrics, parameters, thresholds):
		self.models = models
		self.fairness_metrics = { k:filtered_arguments(v) for k,v in six.iteritems(fairness_metrics) }
		self.performance_metrics = { k:filtered_arguments(v) for k,v in six.iteritems(performance_metrics) }
		self.parameters = parameters
		self.thresholds = thresholds
		self.search_results = {}
		self.pareto_optimal_results = {}

	def _build_grid_param_arrays(self):
			self.n_experiments = 0
			attribute_categories = []

			for key, _ in six.iteritems(self.models):
				model = self.models[key]
				keys, values = zip(*self.parameters[key].items())
				keys = keys + (THRESHOLD_STR, )
				values = values + (self.thresholds, )
				attribute_categories.extend(keys)
				self.n_experiments += len([dict(zip(keys, v)) for v in itertools.product(*values)])
			
			for attribute in list(set(attribute_categories)):
				self.search_results["param_" + attribute] = [np.nan] * self.n_experiments
			
			return

	def _build_score_arrays(self, data):
		scores = {}
	
		for protected_attribute, _ in six.iteritems(data["attribute_map"]):
			for fairness_metric, _ in six.iteritems(self.fairness_metrics):
				self.search_results["score_" + protected_attribute + "_" + fairness_metric] = [np.nan] * self.n_experiments
		for performance_metric, _ in six.iteritems(self.performance_metrics):
			self.search_results["score_" + performance_metric] = [np.nan] * self.n_experiments
		
		return
	
	def _fit_grid(self, data, verbose=False, n_train_samples=None, n_val_samples=None):
		#TODO add verbose functionality
		i = -1
		args_dict = {}

		if n_train_samples:
			train_idx = np.random.choice(data["X_train"].shape[0], n_train_samples, replace=False)
			X_train = data["X_train"][train_idx, :]
			y_train = data["y_train"][train_idx]
		else:
			X_train = data["X_train"]
			y_train = data["y_train"]

		if n_val_samples:
			val_idx = np.random.choice(data["X_val"].shape[0], n_val_samples, replace=0)
			X_val = data["X_val"][val_idx, :]
			y_val = data["y_val"][val_idx]
		else:
			X_val = data["X_val"]
			y_val = data["y_val"]

		args_dict["X"] = X_val
		args_dict["y_true"] = y_val

		for model_key, model_family in six.iteritems(self.models):
			parameter_keys, parameter_values = zip(*self.parameters[model_key].items())
			experiments = [dict(zip(parameter_keys, v)) for v in itertools.product(*parameter_values)]
			for experiment in experiments:    
				# Train Model
				model = model_family(**experiment)
				model = model.fit(X_train, y_train)
				args_dict["y_pred_proba"] = model.predict_proba(X_val)[:, 1]
				args_dict["trained_model"] = model

				for threshold in self.thresholds:
					
					args_dict["threshold"] = threshold
					args_dict["y_pred"] = args_dict["y_pred_proba"] > threshold

					i += 1
					self.search_results["param_threshold"][i] = threshold
					# Fill in parameter values
					for experiment_key, experiment_value in six.iteritems(experiment):
						self.search_results["param_" + experiment_key][i] = experiment_value	

					# Evaluate Model
					for protected_attribute, _ in six.iteritems(data["attribute_map"]):
						args_dict["attribute_map"] = data["attribute_map"][protected_attribute]
						for fairness_metric, fairness_metric_function in six.iteritems(self.fairness_metrics):
							self.search_results["score_" + protected_attribute + "_" + fairness_metric][i] = fairness_metric_function(**args_dict)
					
					for performance_metric, performance_metric_function in six.iteritems(self.performance_metrics):
						self.search_results["score_" + performance_metric][i] = performance_metric_function(**args_dict)

		for key, value in six.iteritems(self.search_results):
			# Hacky way to check for nans, but other ways seemed to break
			mask = [j != j for j in self.search_results[key]]
			self.search_results[key] = ma.array(self.search_results[key], mask=mask)

		self.pareto_optimal_results = self.filter_solution_set()

	def fit(self, data, verbose=1, search_strategy=GRID, n_random_models=None, n_train_samples=None, n_val_samples=None):
		"""
		Based in part on http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/

		Parameters
		----------
		X : 2d array-like 
		Training dataset where rows are instances and columns are features.

		y : 1d array-like 
		Classification labels


		attribute_map : dict of dicts
		denotes the protected attributes of the category of protected 
		attribute (e.g. "Race") to measure causal fairness 
		maps the attribute name to the column and value that correspond 
		to that attribute 
		e.g. one-hot encoding {"Purple": {"col": 0, "val": 1}, 
							   "Green": {"col": 1, "val": 1}}

		e.g. categorical encoding {"Purple": {"col": 0, "val: 1"}, 
								   "Green": {"col": 0, "val: 0"}}                   

		Note: these MUST be mutually exclusive categories!
		
		is_categorical : bool (optional)
		denotes whether the attribute map represents a categorical encoding. If False
		we assume that the encoding is one-hot. 

		max_models : None or int
			If None, return the entire Pareto frontier of models
			Otherwise, return int number of models, ties will be broken randomly  

		search_strategy : str
			'random', a random search over models/hyperparameters
			'grid', enumerates the space of models/hyperparameters
			'genetic_algorithms', uses genetic algorithms 
		"""

		assert search_strategy in SEARCH_STRATEGIES

		if search_strategy == RANDOM:
			assert n_random_models > 0
			#TODO


		if search_strategy == GRID:
			self._build_grid_param_arrays()
			self._build_score_arrays(data)
			self._fit_grid(data, verbose=verbose, n_train_samples=n_train_samples, n_val_samples=n_val_samples)


	def filter_solution_set(self, omitted_score_list=[]):
    	# Inspired by https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
	    assert(self.search_results)

	    costs = -1 * np.array([v for k,v in six.iteritems(self.search_results) if ((k[:5] == "score") & (k[6:] not in omitted_score_list))]).T

	    is_efficient = np.ones(costs.shape[0], dtype = bool)
	    for i, c in enumerate(costs):
	        if is_efficient[i]:
	            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
	    
	    return { k:v[is_efficient] for k,v in six.iteritems(self.search_results)}

