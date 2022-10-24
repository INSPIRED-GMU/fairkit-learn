from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score as accuracy
import math

def classifier_quality_score(model, test_data,
                                 unprivileged_groups,
                                 privileged_groups):
    
    classified_data = model.predict(test_data)
    metric_library = UnifiedMetricLibrary(test_data, classified_data, unprivileged_groups, privileged_groups)

    # call all metrics

    #accuracy

    acc = metric_library.accuracy_score()

    #fairness
    fairness_scores = []
    
    # equal opportunity difference
    eq_opp_diff = metric_library.equal_opportunity_difference()
    fairness_scores.append(eq_opp_diff)

    # average odds difference
    avg_odds_diff = metric_library.average_odds_difference()
    fairness_scores.append(avg_odds_diff)

    # statistical parity difference
    stat_parity_diff = metric_library.statistical_parity_difference()
    fairness_scores.append(stat_parity_diff)
    
    # average odds difference
    avg_odds_diff = metric_library.average_odds_difference()
    fairness_scores.append(avg_odds_diff)

    # calculate & return overall quality score
    max_fair_score = max(fairness_scores)
    balance_val = acc * (1-max_fair_score)

    return math.sqrt(balance_val)
    
    
class UnifiedMetricLibrary():

    def __init__(self, test_data, classified_data, unprivileged_groups, privileged_groups):

        self.test_data = test_data
        self.classified_data = classified_data
        
        self.classification_metric = ClassificationMetric(test_data, classified_data, unprivileged_groups, privileged_groups)

    def accuracy_score(self):
        return accuracy(self.test_data.labels, self.classified_data.labels)

    def num_true_positives(self):
        return self.classification_metric.num_true_positives()

    def num_false_positives(self):
        return self.classification_metric.num_false_positives()

    def num_false_negatives(self):
        return self.classification_metric.num_false_negatives()

    def num_generalized_true_positives(self):
        return self.classification_metric.num_generalized_true_positives()

    def num_generalized_false_positives(self):
        return self.classification_metric.num_generalized_false_positives()

    def num_generalized_false_negatives(self):
        return self.classification_metric.num_generalized_false_negatives()

    def num_generalized_true_negatives(self):
        return self.classification_metric.num_generalized_true_negatives()

    def true_positive_rate(self):
        return self.classification_metric.true_positive_rate()

    def false_positive_rate(self):
        return self.classification_metric.false_positive_rate()

    def false_negative_rate(self):
        return self.classification_metric.false_negative_rate()

    def true_negative_rate(self):
        return self.classification_metric.true_negative_rate()

    def generalized_true_positive_rate(self):
        return self.classification_metric.generalized_true_positive_rate()

    def generalized_false_positive_rate(self):
        return self.classification_metric.generalized_false_positive_rate()

    def generalized_false_negative_rate(self):
        return self.classification_metric.generalized_false_negative_rate()

    def generalized_true_negative_rate(self):
        return self.classification_metric.generalized_true_negative_rate()

    def positive_predictive_value(self):
        return self.classification_metric.positive_predictive_value()

    def false_discovery_rate(self):
        return self.classification_metric.false_discovery_rate()

    def false_omission_rate(self):
        return self.classification_metric.false_omission_rate()

    def negative_predictive_value(self):
        return self.classification_metric.negative_predictive_value()

    def error_rate(self):
        return self.classification_metric.error_rate()

    def false_positive_rate_difference(self):
        return self.classification_metric.false_positive_rate_difference()

    def false_negative_rate_difference(self):
        return self.classification_metric.false_negative_rate_difference()

    def false_omission_rate_difference(self):
        return self.classification_metric.false_omission_rate_difference()

    def false_discovery_rate_difference(self):
        return self.classification_metric.false_discovery_rate_difference()

    def false_positive_rate_ratio(self):
        return self.classification_metric.false_positive_rate_ratio()

    def false_negative_rate_ratio(self):
        return self.classification_metric.false_negative_rate_ratio()

    def false_omission_rate_ratio(self):
        return self.classification_metric.false_omission_rate_ratio()

    def false_discovery_rate_ratio(self):
        return self.classification_metric.false_discovery_rate_ratio()

    def average_abs_odds_difference(self):
        return self.classification_metric.average_abs_odds_difference()

    def error_rate_difference(self):
        return self.classification_metric.error_rate_difference()

    def error_rate_ratio(self):
        return self.classification_metric.error_rate_ratio()

    def num_pred_positives(self):
        return self.classification_metric.num_pred_positives()

    def num_pred_negatives(self):
        return self.classification_metric.num_pred_negatives()

    def selection_rate(self):
        return self.classification_metric.selection_rate()

    def equal_opportunity_difference(self):
        return abs(self.classification_metric.equal_opportunity_difference())

    def average_odds_difference(self):
        return abs(self.classification_metric.average_odds_difference())

    def disparate_impact(self):
        return abs(self.classification_metric.disparate_impact())

    def statistical_parity_difference(self):
        return abs(self.classification_metric.statistical_parity_difference())
