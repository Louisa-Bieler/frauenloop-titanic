### Import packages
# linear algebra
import numpy as np 
# data processing
import pandas as pd 
from sklearn.impute import SimpleImputer
# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split

def split_set(df, size):
    """
    df: Dataframe to split
    size: proportion to data to allocate to validation set (same as train_test_split's test_size)
    """
    return train_test_split(df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'not_alone','Sex_female','Embarked_Q','Embarked_S']], df['Survived'], test_size=size)

X_train, X_test, y_train, y_test = split_set(df,.3)

def predict_kNN(dataset, min_neighbors= 1, max_neighbors = 15, cross_val_folds=5):
    """
    k-NN algoritm and finding the best parameters for it
    """
    param_grid = {'n_neighbors': np.arange(min_neighbors, max_neighbors)}
    knn = KNeighborsClassifier()
    knn_cv = GridSearchCV(knn, param_grid, cv=cross_val_folds)
    knn_cv.fit(X_train, y_train)
    y_pred = knn_cv.predict(X_test)
    print(knn_cv.best_params_)
    print(knn_cv.best_score_)
    print(classification_report(y_test,y_pred))
    knn_classreport= classification_report(y_test,y_pred)
    return knn_classreport

def predict_logreg(dataset):
    """Logistic Regression algorithm"""
    logreg=LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred=logreg.predict(X_test)
    print(classification_report(y_test,y_pred))
    logreg_classresport=classification_report(y_test,y_pred)
    return log_classreport

def predict_RF(dataset):
    """Random Forest  - finding the best hyperparameters"""
    rf=RandomForestClassifier()
    params_rf={'criterion':['gini','entropy'],'n_estimators':[100,200], 'max_depth':[3,4,5], 'min_samples_leaf':[1,2,3,5],'min_samples_split':[2,3,4,5], 'max_features':['auto','log2']}
    grid_rf=GridSearchCV(estimator=rf, param_grid=params_rf,cv=10, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_hyper_rf=grid_rf.best_params_
    print('Best hyperparameters\n', best_hyper_rf)
    print('Best Score: %s' % grid_rf.best_score_)
    return best_hyper_rf

def bestRF(dataset):
    """Running Random Forest algorithm with the best hyperparameters"""
    rf_bestmodel=RandomForestClassifier(criterion='entropy', max_depth=3, max_features='auto', min_samples_leaf=3, min_samples_split=5, n_estimators=100, class_weight='balanced')
    rf_bestmodel.fit(X_train,y_train)
    y_bestpred=rf_bestmodel.predict(X_test)
    rf_classreport=classification_report(y_test,y_bestpred)
    return rf_classreport