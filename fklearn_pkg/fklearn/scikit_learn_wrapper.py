from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC as svc

class ScikitLearnWrapper():

    def __init__(self, model_class, **kwargs):
        self.model = model_class(**kwargs)


    def fit(self, dataset_train):
        self.model.fit(dataset_train.features, dataset_train.labels.ravel())
        
    def predict(self, dataset_test):
        
        dataset_test_pred = dataset_test.copy()
        dataset_test_pred.labels = self.model.predict(dataset_test.features).reshape(-1,1)
        
        return dataset_test_pred


LogisticRegression = lambda **kwargs : ScikitLearnWrapper(lr,**kwargs)
KNeighborsClassifier = lambda **kwargs : ScikitLearnWrapper(knc,**kwargs)
RandomForestClassifier = lambda **kwargs : ScikitLearnWrapper(rfc,**kwargs)
SVC = lambda **kwargs : ScikitLearnWrapper(svc,**kwargs)
