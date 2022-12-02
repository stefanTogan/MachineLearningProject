import os
import sys
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

df_kudos = pd.read_csv('kudos_tiny.csv')
df_baseline = pd.read_csv('recs.csv')
df_large = pd.read_csv('pairs.csv')

def baseline_analysis(df_baseline, df_kudos, n):
    intersection_cols = pd.Index.intersection(df_kudos.columns, df_baseline.columns)
    df_kudos_base = df_kudos[intersection_cols]
    df_kudos_base = df_kudos_base.reindex(sorted(df_kudos_base.columns), axis=1)
    df_baseline = df_baseline.reindex(sorted(df_baseline.columns), axis=1)
    df_copy = df_baseline.copy()
    df_copy = df_copy.drop(['users'], axis=1)
    df_numpy = df_copy.to_numpy()
    thresholds = np.zeros(len(df_copy.index))
    for i in range(len(df_numpy)):
        row = np.sort(df_numpy[i])
        thresholds[i] = row[int((1-n)*len(row))]
    for t in range(len(thresholds)):
        for c in df_copy.columns:
            if df_copy.at[t, c] > thresholds[t]:
                df_copy.at[t, c] = 1
            else:
                df_copy.at[t, c] = 0
    df_kudos_base_copy = df_kudos_base.drop(['users'], axis=1).copy()
    df_base_numpy = df_copy.to_numpy()
    df_kudos_numpy = df_kudos_base_copy.to_numpy()
    recall_scores = []
    for y_pred, y_true in zip(df_base_numpy, df_kudos_numpy):
        recall_scores.append(metrics.precision_score(y_true, y_pred))
    return recall_scores

def plot_baseline(n_set):
    fig, ax = plt.subplots()
    mean_score = []
    std_score = []
    for n in n_set:
        scores = baseline_analysis(df_baseline, df_kudos, n)
        mean_score.append(np.mean(scores))
        std_score.append(np.std(scores))
        print(np.mean(scores))
    ax.errorbar(n_set, mean_score, yerr = std_score, linewidth = 2, color='hotpink', label = 'Precision Score')
    ax.set_ylabel('Precision Score')
    ax.set_xlabel('Threshold to Recommend')
    ax.legend()

def logreg_analysis(model, X, y):
    kf = StratifiedKFold(n_splits=5)
    #maybe add interaction terms
    temp_acc = []
    temp_recall = []
    temp_precision = []
    temp_f1 = []
    for train, test in kf.split(X, y):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        temp_acc.append(metrics.accuracy_score(y[test], y_pred))
        temp_recall.append(metrics.recall_score(y[test], y_pred))
        temp_precision.append(metrics.precision_score(y[test], y_pred))
        temp_f1.append(metrics.f1_score(y[test], y_pred))
        print(confusion_matrix(y[test], y_pred))
    return [temp_acc, temp_recall, temp_precision, temp_f1]

def logreg_plot(X,y):
    C_set = [0.01, 0.1, 0.5, 1, 10, 100]
    fig, ax = plt.subplots()
    fig.suptitle('Precision of Logistic Regression Models')
    mean_score = []
    std_score = []
    for C in C_set:
        model_ridge = LogisticRegression(penalty = 'l2', solver = 'lbfgs', C=C, max_iter = 10000)
        score = logreg_analysis(model_ridge, X, y)
        mean_score.append(np.array(score).mean())
        std_score.append(np.array(score).std())

    ax.errorbar(C_set, mean_score, yerr = std_score, linewidth = 2, color='hotpink', label = 'Ridge')
    model_logit = LogisticRegression(penalty = 'none', solver = 'lbfgs', max_iter = 10000)
    score = logreg_analysis(model_logit, X, y)
    ax.axhline(y = np.array(score).mean(), color = 'green', linestyle='-', label = 'Logit')
    ax.set_ylabel('Precision Score')
    ax.set_xlabel('C')
    ax.legend()

def KNNcrossval(X, y):
    kf = StratifiedKFold(n_splits=4)
    plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
    mean_acc=[]; std_acc=[]
    mean_recall=[]; std_recall=[]
    mean_precision=[]; std_precision=[]
    mean_f1=[]; std_f1=[]
    k_range = range(2,18,3)
    fig, ax = plt.subplots()
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        temp_acc=[]
        temp_recall=[]
        temp_precision=[]
        temp_f1=[]
        for train,test in kf.split(X, y):
            knn.fit(X[train], y[train])
            ypred = knn.predict(X[test])
            temp_acc.append(metrics.accuracy_score(y[test],ypred))
            temp_recall.append(metrics.recall_score(y[test], ypred))
            temp_precision.append(metrics.precision_score(y[test], ypred))
            temp_f1.append(metrics.f1_score(y[test], ypred))
        mean_acc.append(np.array(temp_acc).mean()); std_acc.append(np.array(temp_acc).std())
        mean_recall.append(np.array(temp_recall).mean()); std_recall.append(np.array(temp_recall).std())
        mean_precision.append(np.array(temp_precision).mean()); std_precision.append(np.array(temp_precision).std())
        mean_f1.append(np.array(temp_f1).mean()); std_f1.append(np.array(temp_f1).std())
        print("k= " + str(k))
        print([np.array(temp_acc).mean(), np.array(temp_recall).mean(), np.array(temp_precision).mean(),
               np.array(temp_f1).mean()])
        print(confusion_matrix(y[test], ypred))
    ax.errorbar(k_range, mean_acc, yerr=std_acc, color='olivedrab', linewidth = 2, label = 'Accuracy')
    ax.errorbar(k_range, mean_recall, yerr=std_recall, color='hotpink', linewidth = 2, label= 'Recall')
    ax.errorbar(k_range, mean_precision, yerr=std_precision, color='goldenrod', linewidth=2, label='Precision')
    ax.errorbar(k_range, mean_f1, yerr=std_f1, color='salmon', linewidth=2, label='F1 Score')
    ax.set_xlabel('#Neighbours (k)')
    ax.set_ylabel('Score')
    ax.legend()

def MLPcrossval(X,y):
    mean_acc = [];
    std_acc = []
    mean_recall = [];
    std_recall = []
    mean_precision = [];
    std_precision = []
    mean_f1 = [];
    std_f1 = []
    fig, ax = plt.subplots()
    C_range = [0.1, 1, 5, 10, 50, 100]
    for C in C_range:
        print('C: ' + str(C))
        model = MLPClassifier(hidden_layer_sizes=(50), alpha=1 / C, max_iter=200)
    #hidden_layer_range = [5, 10, 25, 50, 75, 100]
    #for n in hidden_layer_range:
    #    print('Hidden layer size: ' + str(n))
    #    model = MLPClassifier(hidden_layer_sizes=(n), max_iter = 300)
        scores = logreg_analysis(model, X, y)
        mean_acc.append(np.array(scores[0]).mean());
        std_acc.append(np.array(scores[0]).std())
        mean_recall.append(np.array(scores[1]).mean());
        std_recall.append(np.array(scores[1]).std())
        mean_precision.append(np.array(scores[2]).mean());
        std_precision.append(np.array(scores[2]).std())
        mean_f1.append(np.array(scores[3]).mean());
        std_f1.append(np.array(scores[3]).std())
    ax.errorbar(hidden_layer_range, mean_acc, yerr = std_acc, linewidth=2, color='olivedrab', label='Accuracy')
    ax.errorbar(hidden_layer_range, mean_recall, yerr=std_recall, linewidth=2, color='hotpink', label='Recall')
    ax.errorbar(hidden_layer_range, mean_precision, yerr=std_precision, linewidth=2, color='goldenrod', label='Precision')
    ax.errorbar(hidden_layer_range, mean_f1, yerr=std_f1, linewidth=2, color='salmon', label='F1 Score')
    #ax.set_xlabel('Hidden Layer Size')
    ax.set_xlabel('C')
    ax.set_ylabel('Score')
    ax.legend()

def MLPsolve(X,y,layers):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MLPClassifier(hidden_layer_sizes=(layers)).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    y_dummy = dummy.predict(x_test)
    print(confusion_matrix(y_test, y_dummy))

    probs = model.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, probs[:,1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label = 'MLP')
    model_ridge = LogisticRegression(penalty='l2', solver='lbfgs', C=0.001, max_iter=10000).fit(x_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, model_ridge.decision_function(x_test))
    ax.plot(fpr, tpr, color='orange', label = 'Logistic Regression')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.plot([0,1], [0,1], color='green', linestyle='--')
    ax.legend()


y = df_large.iloc[:,2].values
X = df_large.iloc[:,3:155].values
n_set = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
#plot_baseline(n_set)
#logreg_plot(X,y)
#KNNcrossval(X,y)
#MLPcrossval(X,y)
MLPsolve(X,y,50)
plt.show()



