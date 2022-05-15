from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_train_and_label_data(df):
    """to use this function, call as follows:
     train, label = get_train_and_label_data(clean_df)

     the function drops the 'Survived' col and value as train and
     returns the values of 'Survived' col as label """

    train = df.drop(columns=['Survived']).values
    label = df['Survived'].values
    return train, label

def get_feature_cols(df):
    """to use this function, call as follows:
         feature_cols = get_feature_cols(clean_df)

         the function drops the 'Survived' col returns the remaining cols """
    return df.drop(columns=['Survived']).columns

def model_decision_tree(df):
    """to use this function, call as follows:
        model_decision_tree(clean_df)
        the func takes a clean df and plot a decision tree    """
    train_data, train_label = get_train_and_label_data(df)
    feature_columns=df.drop(columns=['Survived']).columns


    tree_clf = DecisionTreeClassifier(max_depth=4)
    tree_clf.fit(train_data, train_label)

    fig = plt.figure(figsize=(27, 27))  # setplot size (denoted in inches)
    _ = tree.plot_tree(tree_clf,
                       max_depth=4,
                       feature_names=feature_columns,
                       class_names={0: 'Died', 1: 'Survived'},
                       label='all',
                       filled=True,
                       rounded=True,
                       precision=2,
                       fontsize=12)
    plt.show()
    fig.savefig('tree_clf.png')

