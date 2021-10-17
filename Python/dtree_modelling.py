# -*- coding: utf-8 -*-
"""
Backend of the project. This will make the dtree model and generate the plot
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,\
                         plot_tree
                         

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_PATH = '../Data/Telco data TC fix.csv'
TREE_DEPTH = 3
PROB = 'regression'
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
UNIQUE_THRESH = 0.5
MAX_CLASSES = 10

class_names = None

# =============================================================================
# FUNCTIONS
# =============================================================================
def loadData(data_path):
    '''
    TEMP: pandas load csv
    FUTURE: streamlit pass thru
    '''
    df = pd.read_csv(DATA_PATH)
    
    return df

def getColumns(df):
    '''
    TEMP: pandas cols list
    FUTURE: feed back to streamlit app
    '''
    cols = list(df.columns)
    
    return cols

def pickResponse(cols):
    '''
    TEMP: input
    FUTURE: streamlit drop down
    '''
    selection = pd.Series(cols)
    print(selection)
    print(f'\nChoose response variable 0-{len(selection)-1}')
    chosen_int = int(input())
    
    response = selection.iloc[chosen_int]
    print(response)
    
    return response
    
def catOrNum(df, NUMERICS):
    '''
    Determine which columns are categorical and which are not
    '''
    dtype_dict = {}
    dtype_dict['num'] = list(df.select_dtypes(include=NUMERICS).columns)
    dtype_dict['cat'] = list(df.select_dtypes(exclude=NUMERICS).columns)
    
    return dtype_dict

def getClassNames(df, response, MAX_CLASSES):
    '''
    Get the class names of a classification problem
    '''
    if PROB == 'regression':
        class_names = None
    else:
        class_names = list(df[response].unique())
        
        if len(class_names) > MAX_CLASSES:
            print(f'\nResponse variable contains over {MAX_CLASSES} classes.',
                  f'\nPlease limit you response variable to {MAX_CLASSES}',
                   'or less.')
    
    return class_names

def dropUniques(df, dtype_dict, UNIQUE_THRESH):
    '''
    drop categorical columns that have too many unique values.
    UNIQUE_THRESH: % of col that must not be unique
    '''
    for col i dtype_dict['cat']:
    
    return df

def processData(df):
    '''
    Combines preprocessing functions to make data ready for modelling
    '''
    return df

def trainTree(df, PROB, response):
    '''
    train a dtree model
    '''
    #regression or classification
    if PROB == 'regression':
        dtree = DecisionTreeRegressor(max_depth=TREE_DEPTH)
        
    elif PROB == 'classification':
        dtree = DecisionTreeClassifier(max_depth=TREE_DEPTH,\
                                       class_weight='balanced')
        
    else:
        print('\n\n\n!!! PROB must be regression or classification !!!\n\n\n')
        
    #train
    dtree.fit(df.drop(response, axis=1), df[response])
    
    return dtree

def genTree(dtree):
    '''
    generates (& displays) a drawn dtree
    '''
    plt.figure(figsize=(20, 12)) 
    plot_tree(dtree, feature_names=df.drop(response, axis=1).columns,\
              class_names=class_names, filled=True, rounded=True, precision=2,\
              proportion=True)
        
# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
    df = loadData(DATA_PATH)
    
    cols = getColumns(df)
    response = pickResponse(cols)
    class_names = getClassNames(df, response)
    dtype_dict = catOrNum(df, NUMERICS)
    print('\n', dtype_dict, '\n')
# =============================================================================
#bike
#     df = df.drop(['casual',
#  'registered',
#  'count cox',
#  'casual cox',
#  'registered cox'], axis=1)
# =============================================================================
    
    dtree = trainTree(df, PROB, response)
    genTree(dtree)
    
    