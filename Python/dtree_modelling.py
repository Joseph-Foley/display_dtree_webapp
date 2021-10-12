# -*- coding: utf-8 -*-
"""
Backend of the project. This will make the dtree model and generate the plot
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# =============================================================================
# CONSTANTS
# =============================================================================
data_path = '../Data/Bike share data (Featurised).csv'

# =============================================================================
# FUNCTIONS
# =============================================================================
def loadData(data_path):
    '''
    TEMP: pandas load csv
    FUTURE: streamlit pass thru
    '''
    df = pd.read_csv(data_path)
    
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
    

# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
    df = loadData(data_path)
    cols = getColumns(df)
    response = pickResponse(cols)
    
    