# Load, Explore, Clean/Transform

# imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
import seaborn as sns
# set the color palette
sns.set_palette(sns.color_palette("husl"))
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn


# load and explore:
def load_explore(csv_file):
    df = pd.read_csv(csv_file, index_col='PassengerId')
    print(df.head(), df.shape, df.dtypes, df.describe(), df.isna().sum() * 100 / len(df))
    return df

df = load_explore('train.csv')


# Transform by making a gender boolean column:

def gender_boolean(row):
        if 'female' in str(row):
            return 1
        else:
            return 0

df['GenderBoolean'] = df['Sex'].apply(gender_boolean)
    
    
def create_clean_df(df, columns_to_keep):
        clean_df = pd.DataFrame()
        for i in columns_to_keep:
            clean_df[i] = df[[i]]
        return clean_df    
    
clean_df = create_clean_df(df, ['Survived', 'Pclass', 'Age', 'SibSp', 'GenderBoolean'])
print(clean_df.head())
    