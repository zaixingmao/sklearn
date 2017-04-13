import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics, preprocessing
import pandas as pd
import math 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

#******************************************************
# the purpose of this code is to: 
# 1) simplify the calling of classification methods
# 2) preform training and return test scores
#******************************************************


def methods(method):
    MLs = {}
    MLs["LR_l2"] = LogisticRegression(C=1000, penalty='l2', tol=0.01)
    MLs["LR_l1"] = LogisticRegression(C=1000, penalty='l1', tol=0.01)
    MLs["BDT"] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=100)
    MLs["LinearSVC"] = LinearSVC()
    MLs["SVC"] = SVC()
    MLs["MLP"] = MLPClassifier(hidden_layer_sizes=(3,5,5,3), max_iter=50, alpha=1e-4,
                    solver='lbfgs', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.5)
    MLs["KNN"] = KNeighborsClassifier(3)

    return MLs[method]    

def method_selection(method, X_train, Y_train, X_test):
    ML = methods(method)
    ML.fit(X_train, Y_train)
    if hasattr(ML, "decision_function"):
        return (ML.decision_function(X_test), ML.predict(X_test))
    else:
        return (ML.predict_proba(X_test)[:,1], ML.predict(X_test))

