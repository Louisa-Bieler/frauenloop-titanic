import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
# set the color palette
sns.set_palette(sns.color_palette("husl"))

def show_feature_hist(df):
    """to use this function, call as follows:
        show_feature_hist(df) """
    df.hist(bins=50, figsize=(20, 15))
    plt.show()

def plot_survived(df, ft):
    """to use this function, call as follows:
        plot_survived(df, df['Survived']) """
    # n° of survivors
    labels = ['Not survived', 'Survived']
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x=ft, ax=ax)
    ax.set_title('Survived')
    plt.xticks(rotation=45, ha='right')
    ax.set_xticklabels(labels)
    ax.set(xlabel=None)

def plot_survived_vs_sib_spouse(df, ft1, ft2):
    """to use this function, call as follows:
        plot_survived_vs_sib_spouse(df, df['Survived'], df['SibSp']) """
    # survived vs n° siblings- spouse per passenger
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(data=df, x=ft1, y=ft2, ax=ax)
    ax.set_title('Survived vs N° siblings-spouse per passenger')
    plt.xticks(rotation=45, ha='right')

def get_correlation(df, ft_name):
    """to use this function, call as follows:
       get_correlation(df, 'Survived') """
    corr_matrix = df.corr()
    return corr_matrix[ft_name].sort_values(ascending=False)

def get_1hot_array(col):
    """to use this function, call as follows:
        one_hot_array = get_1hot_array(df[['Embarked]]) """
    cat_encoder = OneHotEncoder()
    col_1hot = cat_encoder.fit_transform(col)
    return col_1hot.toarray()