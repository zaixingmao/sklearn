import matplotlib.pyplot as plt
import numpy as np
#!/usr/bin/env python
import collections
from sklearn import datasets, linear_model, metrics, preprocessing
import pandas as pd
import math 
from sklearn.linear_model import LogisticRegression
from classifier_definitions import method_selection
from sklearn.model_selection import train_test_split

#******************************************************
# the purpose of this code is to: 
# 1) load the cleaned data and perform feature scaling; 
# 2) split the data into testing and training sets; 
# 3) perform classification tasks;
# 4) compare performance metrics
#******************************************************

def main():
    cleaned_df = pd.read_csv('/Users/zmao/M-Data/School/scipy/cleaned_data.csv')
    npMatrix = np.matrix(cleaned_df)
    X = npMatrix[:, :-1]
    Y = npMatrix[:, -1:]
    Y, = np.array(Y.T)
    column_names = cleaned_df.columns

    #scale data
    X_s = preprocessing.StandardScaler().fit_transform(X)
    X_r = preprocessing.RobustScaler().fit_transform(X)

    #plot the effects of scaling
    n_features = len(cleaned_df.columns) -1
    n_rows = (math.ceil(n_features/2.))
    fig, ax = plt.subplots(n_rows, 3, figsize=(12, 4*n_rows))
    for i in range(0, n_rows):
        next = 2*i+1 if 2*i+1 < n_features else 0
        ax[i, 0].scatter(X[:, 2*i], X[:, next])
        ax[i, 1].scatter(X_s[:, 2*i], X_s[:, next])
        ax[i, 2].scatter(X_r[:, 2*i], X_r[:, next])

        ax[i, 0].set_title("Unscaled data")
        ax[i, 1].set_title("After standard scaling")
        ax[i, 2].set_title("After robust scaling")
        for j in range(3):
            ax[i, j].set_xlabel(column_names[2*i])
            ax[i, j].set_ylabel(column_names[next])

    plt.tight_layout()
    plt.savefig("/Users/zmao/M-Data/School/scipy/features/feature_scaling.png")
    plt.figure()  # New window

    #split data into training and testing
    test_size = 0.25
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    X_s_train, X_s_test, Y_s_train, Y_s_test = train_test_split(X_s, Y, test_size=test_size, random_state=0)
    X_r_train, X_r_test, Y_r_train, Y_r_test = train_test_split(X_r, Y, test_size=test_size, random_state=0)

    #training ********
    methods = ['MLP', 'LR_l1', 'LR_l2', 'BDT', 'LinearSVC', 'SVC', 'KNN']
    for method in methods:
        res_dict = collections.defaultdict(list)

        res_dict[method].append(method_selection(method, X_train, Y_train, X_test))
        res_dict[method].append(method_selection(method, X_s_train, Y_s_train, X_s_test))
        res_dict[method].append(method_selection(method, X_r_train, Y_r_train, X_r_test))

        #results **********
        fpr, tpr, _ = metrics.roc_curve(Y_test, res_dict[method][0][0])
        fpr_s, tpr_s, _ = metrics.roc_curve(Y_s_test, res_dict[method][1][0])
        fpr_r, tpr_r, _ = metrics.roc_curve(Y_r_test, res_dict[method][2][0])

        auc = metrics.roc_auc_score(Y_test, res_dict[method][0][0])
        auc_s = metrics.roc_auc_score(Y_s_test, res_dict[method][1][0])
        auc_r = metrics.roc_auc_score(Y_r_test, res_dict[method][2][0])

        acc = metrics.accuracy_score(Y_test, res_dict[method][0][1])
        acc_s = metrics.accuracy_score(Y_s_test, res_dict[method][1][1])
        acc_r = metrics.accuracy_score(Y_r_test, res_dict[method][2][1])

        
        print("AUC: %.3f" %auc)

        #plotting ******
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='(Unscaled features) AUC = %.3f, acc = %.3f' %(auc, acc))
        plt.plot(fpr_s, tpr_s, color='blue', lw=2, label='(Standard scaled features) AUC = %.3f, acc = %.3f' %(auc_s, acc_s))
        plt.plot(fpr_r, tpr_r, color='red', lw=2, label='(Robust scaled features) AUC = %.3f, acc = %.3f' %(auc_r, acc_r))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        plt.savefig("/Users/zmao/M-Data/School/scipy/ML_results/%s_ROC_curve.png" %method)
        plt.figure()  # New window


if __name__ == "__main__":
    main()