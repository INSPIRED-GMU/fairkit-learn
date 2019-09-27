"""
Basic IO for loading fair datasets 

Ex: http://scikit-learn.org/stable/datasets/twenty_newsgroups.html#newsgroups

TODO: should we have "vectorized" and "non-vectorized" versions here like sklearn does?
"""
from __future__ import division
import warnings
import numpy as np 
import sklearn
import pandas as pd  

def fetch_adult_income():
    #directly downloads the data 
    #calls load to load the data 
    #TODO 
    pass

def load_adult_income_train_val():
    #for the user study so they can't access the test set 
    return load_adult_income(train_val_split=0.5, notest=True)

def load_adult_income(train_val_split=0.5, notest=False):
    """
    Load files and data from the propublica dataset 

    Parameters
    ----------
    train_val_split : float
        Amount to split the training set to create a train and validation set 

    Returns
    -------
    data : dict
        With keys

        X : 2-d ndarray 

        y : 1-d ndarray

        (or X_train, y_train, X_test, y_test if subset=='all')

        feat_names : list of strs
            List of the feature names corresponding to the indices of the columns
            of X  

        attribute_map : dict of dicts of dicts 
            Denotes the protected attributes of the category of protected 
            attribute (e.g. "Race") and maps the attribute name to the column and value that correspond 
            to that attribute 
            e.g. one-hot encoding for a one-hot encoding denoting the columns ("col") and values ("val")
            
             {"Race": {"Purple": {"col": 0, "val": 1}, "Green": {"col": 1, "val": 1}},
                                   "Sex" : {"Female": {"col": 0, "val": 1}, "Male": {"col": 1, "val": 1}}

            e.g. categorical encoding {"Purple": {"col": 0, "val: 1"}, 
                                       "Green": {"col": 0, "val: 2"}}                   

            Note: these MUST be mutually exclusive categories! 

        is_categorical : boolean 
            True if the y-values are categorical 
            False otherwise (indicating a one-hot encoding) 

    Examples
    --------
    >>> from fklearn.datasets import load_adult_income
    """
    data = {}
    data['is_categorical'] = False
    header_names = ["Age", "Workclass", "FNLWGT", "Education", "Education-Num", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per Week", "Native Country", "Income"]
    
    train_ref = pd.read_csv("../data/adult_income/train.csv", index_col = False, delimiter=' *, *', engine='python', names = header_names)
    train_all = pd.get_dummies(train_ref)
    train_all.columns = train_all.columns.str.replace('_ ', '_')

    end_idx_train_val_split = int(np.floor(train_val_split*train_all.shape[0]))
    train = train_all[:end_idx_train_val_split]
    val = train_all[end_idx_train_val_split:]

    y_train =  train["Income_<=50K"].copy()
    X_train = train.drop(["Income_<=50K","Income_>50K", "Native Country_Holand-Netherlands"], axis=1).copy()
    y_val =  val["Income_<=50K"].copy()
    X_val = val.drop(["Income_<=50K","Income_>50K", "Native Country_Holand-Netherlands"], axis=1).copy()
    
    test_ref = pd.read_csv( "../data/adult_income/test.csv", index_col = False, delimiter=' *, *', engine='python', names = header_names)
    test = pd.get_dummies(test_ref)
    test.columns = test.columns.str.replace('_ ', '_')
    y_test = test["Income_<=50K."].copy()
    X_test = test.drop(["Income_<=50K.","Income_>50K."], axis=1).copy()

    data['feat_names'] = [str(col) for col in X_test.columns] 

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    data['X_train'] = X_train.values
    data['y_train'] = y_train.values
    data['X_val'] = X_val.values
    data['y_val'] = y_val.values
    data['X_test'] = X_test.values
    data['y_test'] = y_test.values

    attribute_map = {'Race': {}, 'Sex': {}}
    for ii, col in enumerate(X_train):
        if col.startswith('Race'):
            attribute_map['Race'][col] = {'col': ii, 'val': 1}
        elif col.startswith('Sex'):
            attribute_map['Sex'][col] = {'col': ii, 'val': 1}
    data['attribute_map'] = attribute_map

    if notest: 
        del data['X_test']
        del data['y_test']

    unprocessed_train_data = train_ref

    return data, unprocessed_train_data

def fetch_propublica(subset='train'):
    """
    Load files and data from the propublica dataset 

    Parameters
    ----------
    subset : 'train' or 'test', 'all'
        Select which dataset to load 

    Returns
    -------
    X : 2-d ndarray 

    y : 2-d ndarray 

    attribute_map : dict of dicts
        Denotes the protected attributes of the category of protected 
        attribute (e.g. "Race") to measure causal fairness 
        maps the attribute name to the column and value that correspond 
        to that attribute 
        e.g. one-hot encoding {"Purple": {"col": 0, "val": 1}, 
                               "Green": {"col": 1, "val": 1}}

        e.g. categorical encoding {"Purple": {"col": 0, "val: 1"}, 
                                   "Green": {"col": 0, "val: 2"}}                   

        Note: these MUST be mutually exclusive categories! 

    is_categorical : boolean 
        True if the y-values are categorical 
        False otherwise (indicating a one-hot encoding) 


    Examples
    --------
    >>> from fklearn.datasets import fetch_propublica 

    """
    pass

if __name__ == '__main__':
    load_adult_income()

#TODO: other dataset functions 

