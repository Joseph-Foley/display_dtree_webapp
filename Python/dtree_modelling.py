# -*- coding: utf-8 -*-
"""
Backend of the project. This will make the dtree model and generate the plot
"""
# =============================================================================
# IMPORTS
# =============================================================================
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,\
                         plot_tree, export_graphviz
                         
from io import BytesIO, StringIO
from PIL import Image

from IPython.display import Image as ImageIpy
import pydot 
                         
# =============================================================================
# DEMO DATA
# =============================================================================
#DATA_PATH = '../Data/Telco data TC fix_resp_int.csv'
#DATA_PATH = '../Data/Telco data TC fix.csv'
#DATA_PATH = '../Data/Bike share data (atemp-weather fix).csv'
DATA_PATH = '../Data/School_Attendance.csv'
#DATA_PATH = '../Data/T20 International Dataset_SMALL.csv'
#DATA_PATH = '../Data/titanic_data.csv'
#DATA_PATH = '../Data/us2021census.csv'

# =============================================================================
# CONSTANTS
# =============================================================================
#PROB = 'Classification'
PROB = 'Regression'
TREE_DEPTH = 3
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
UNIQUE_THRESH = 0.2
MAKE_NUM_THRESH = 0.95
MAX_CLASSES = 10
CAT_LIMIT = 10
COL_LIMIT = 100
SEP = '$!@'

#assert MAX_CLASSES <= CAT_LIMIT

# =============================================================================
# FUNCTIONS
# =============================================================================
def loadData(DATA_PATH):
    '''
    TEMP: pandas load csv
    FUTURE: streamlit pass thru
    '''
    df = pd.read_csv(DATA_PATH, encoding_errors='ignore')
    
    return df

def getColumns(df):
    '''
    TEMP: pandas cols list
    FUTURE: feed back to streamlit app
    '''
    cols = list(df.columns)
    
    return cols

def checkColLimit(cols, COL_LIMIT):
    '''
    Number of columns must not exceed limit
    '''
    if len(cols) > COL_LIMIT:
        note = '\nCOLUMN LIMIT EXCEEDED' + \
               f'\nPLEASE LIMIT YOUR DATA TO {COL_LIMIT} COLUMNS'
        
        print(note)
        
        return note

def pickResponse(cols):
    '''
    Back end only. Simulates user input of response variable.
    '''
    selection = pd.Series(cols)
    print(selection)
    print(f'\nChoose response variable 0-{len(selection)-1}')
    chosen_int = int(input())
    
    response = selection.iloc[chosen_int]
    print(response)
    
    return response

def makeRespStr(df, response, PROB):
    '''
    Classification response must be string type
    '''
    if PROB == 'Classification':
        df[response] = df[response].astype(str)
        
    else:
        pass
    
    return df

def getClassNames(df, response, MAX_CLASSES, PROB):
    '''
    Get the class names of a Classification problem
    '''
    if PROB == 'Regression':
        class_names = None
    else:
        class_names = list(df[response].unique())
        
        if len(class_names) > MAX_CLASSES:
            note = f'\nResponse variable contains over {MAX_CLASSES} classes'+\
                   f'\nPlease limit you response variable to {MAX_CLASSES} '+\
                   'or less.\nAre you sure this is not a Regression problem?'
            
            print(note)
            
            return note
            
        class_names.sort()
    
    return class_names
    
def catOrNum(df, NUMERICS):
    '''
    Determine which columns are categorical and which are not
    '''
    dtype_dict = {}
    dtype_dict['num'] = list(df.select_dtypes(include=NUMERICS).columns)
    dtype_dict['cat'] = list(df.select_dtypes(exclude=NUMERICS).columns)
    
    return dtype_dict

def processNulls(df, dtype_dict):
    '''
    substitue values for null cells
    '''
    #median for numerical
    for col in dtype_dict['num']:
        df[col] = df[col].fillna(round(df[col].median(),0))
        
    #'NULL' for cat
    for col in dtype_dict['cat']:
        df[col] = df[col].fillna('NULL')
        
    return df

def makeNumeric(df, dtype_dict, MAKE_NUM_THRESH):
    '''
    If col has more numeric values than non-numeric then make it numeric
    MAKE_NUM_THRESH: % of rows that are actually numeric
    '''
    for col in dtype_dict['cat']:
        #convert bool cols to str
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
        
        #find strings, if item is not a string then it will return NaN
        num_bool = df[col].str.contains('').isna()
        
        #Check thresh
        if num_bool.sum() / len(df) > MAKE_NUM_THRESH:
            
            #remove string rows
            df = df[num_bool]
            
            #change what is recorded in dict
            dtype_dict['num'].append(col)
            dtype_dict['cat'].remove(col)
            
    return df, dtype_dict
            
def checkRegResponse(df, response, dtype_dict, PROB):
    '''
    if there still too many non numerical columns in Regression response then
    terminate.
    '''
    if PROB == 'Regression'\
    and response in dtype_dict['cat']:
        
        note = '\nResponse variable has too many non numerical values,\n'+\
               'Are you sure this is a Regression problem? \n'+\
               'If so then edit the response column so that it only contains '+\
               'numerical values\n'
        
        print(note)
        
        return note
        
def dropUniques(df, dtype_dict, UNIQUE_THRESH):
    '''
    drop categorical columns that have too many unique values.
    UNIQUE_THRESH: % of col that must not be unique
    '''
    cat_cols = dtype_dict['cat'].copy()
    dropped = []
    for col in cat_cols:       
        if df[col].nunique()/len(df) > UNIQUE_THRESH:
            df = df.drop(col, axis=1)
            dropped.append(col)
            
            dtype_dict['cat'].remove(col)
            
    if len(dropped) > 0:
        print('The following columns have been dropped as they contained too',
              f'many unique categories: \n{dropped}',
              '\nWere they meant to be categorical?\n')
    
    
    return df, dtype_dict, dropped

def limitCats(df, dtype_dict, CAT_LIMIT):
    '''
    Categorical variables can only have up to CAT_LIMIT categories. Uncommon
    categories will be badged as "AAAOTHER"
    '''
    for col in dtype_dict['cat']:
        #check if exceeds limit
        if df[col].nunique() <= CAT_LIMIT - 1:
            continue
        
        else:
            #find least frequent cats
            val_counts = df[col].value_counts()#.sort_index(ascending=False)
            
            #change name of least frequent cats
            least_freq = list(val_counts.iloc[CAT_LIMIT - 1:].index)
            df[col][df[col].isin(least_freq)] = 'AAAOther'
            
    return df

def ordinalResponse(df, response, class_names, dtype_dict, PROB):
    '''
    response var in categorical problems needs to be in ordinal format
    e.g. red, green, blue => 0, 1, 2
    '''
    if PROB == 'Regression':
        pass
    
    elif PROB == 'Classification':
        #make map
        ord_map = pd.Series(index=class_names,\
                            data=range(len(class_names)))
            
        #apply map
        df[response] = df[response].replace(ord_map)
            
        #remove from cats
        dtype_dict['cat'].remove(response)
        dtype_dict['num'].append(response)
        
    else:
        print('\n\n\n!!! PROB must be Regression or Classification !!!\n\n\n') 
        
        
    return df, dtype_dict
        
def dummyVars(df, dtype_dict, SEP):
    '''
    create dummy variables from categoricals.
    AAAOther and first object in alphabetical order to be dropped. (helps 
    remove "No" in Yes/No columns)
    '''
    return pd.get_dummies(df, columns=dtype_dict['cat'], drop_first=True,\
                          prefix_sep=SEP)
        
def processData(df):
    '''
    Combines preprocessing functions to make data ready for modelling
    checkRegResponse might need to be outside this tho
    '''
    return df

def trainTree(df, PROB, response):
    '''
    train a dtree model
    '''
    #Regression or Classification
    if PROB == 'Regression':
        dtree = DecisionTreeRegressor(max_depth=TREE_DEPTH)
        
    elif PROB == 'Classification':
        dtree = DecisionTreeClassifier(max_depth=TREE_DEPTH)
        
    else:
        print('\n\n\n!!! PROB must be Regression or Classification !!!\n\n\n')
        
    #train
    dtree.fit(df.drop(response, axis=1), df[response])
    
    return dtree

def genTree(df, dtree, class_names, response, PROB,\
            w=14, h=6, dpi=300, fontsize=8):
    '''
    generates (& displays) a drawn dtree
    WILL REMOVE
    '''
    #TODO test figsize on variety of screens
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (w, h), dpi=dpi)
    plot_tree(dtree, feature_names=df.drop(response, axis=1).columns,\
              class_names=class_names, filled=True, rounded=True, precision=2,\
              proportion=True, impurity=False, fontsize=fontsize)
    
    axes.title.set_text(f'{PROB} Decision Tree for {response}')
    #save as image to memory
    mem_fig = BytesIO()
    fig.savefig(mem_fig)
    
    print('\nTree nodes looking at bit small?',\
          '\nTry renaming you columns with less characters\n')
        
    return mem_fig

def genTreeGV(df, dtree, class_names, response, PROB, dpi=300):
    '''
    generates (& displays) a drawn dtree using pydot and graphviz
    '''
    #export to gv
    dot_data = StringIO()  
    export_graphviz(dtree, out_file=dot_data,\
                    feature_names=df.drop(response, axis=1).columns,\
                    class_names=class_names, filled=True, rounded=True,\
                    precision=2, proportion=True, impurity=False)
    
    #create png and save to memory
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
    graph.set_dpi(dpi)
    mem_fig_gv = BytesIO(graph.create_png())
    
    return mem_fig_gv
        
# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
    #load df
    df = loadData(DATA_PATH)
    
    #get column names
    cols = getColumns(df)
    
    #user picks response column
    response = pickResponse(cols)
    
    #make response col dtype string if Classification
    df = makeRespStr(df, response, PROB)
    
    #get class names if Classification
    class_names = getClassNames(df, response, MAX_CLASSES, PROB)
    if type(class_names) is str:
        sys.exit()
    
    #determine categorical and numerical columns
    dtype_dict = catOrNum(df, NUMERICS)
    print('\n', dtype_dict, '\n')
    
    #process null values
    df = processNulls(df, dtype_dict)
    
    #make column numeric if most values are
    df, dtype_dict = makeNumeric(df, dtype_dict, MAKE_NUM_THRESH)
    
    #if too many values in reg response are categorical then terminate and warn
    note = checkRegResponse(df, response, dtype_dict, PROB)
    if note is not None:
        sys.exit()
    
    #drop categorical columns if there are too many unique values
    df, dtype_dict, dropped = dropUniques(df, dtype_dict, UNIQUE_THRESH)
    
    #limit number of categories in cat columns
    df = limitCats(df, dtype_dict, CAT_LIMIT)
    
    #if Classification then ordinal encode the response (for sklearn)
    df, dtype_dict = ordinalResponse(df, response, class_names, dtype_dict, PROB)
    
    #one hot encode the categorical columns
    df = dummyVars(df, dtype_dict, SEP)
    
    #train a tree
    dtree = trainTree(df, PROB, response)
    
# =============================================================================
#     #generate the tree graphic to BytesIO
#     mem_fig = genTree(df, dtree, class_names, response, PROB,\
#                       w=14, h=6, dpi=125, fontsize=8)
#     
#     #render in console
#     mem_fig.seek(0)
#     image_plot = Image.open(mem_fig)
# =============================================================================
    
    #graphviz tree image
    mem_fig_gv = genTreeGV(df, dtree, class_names, response, PROB, dpi=90)
    
    image_plot = Image.open(mem_fig_gv)
    image_plot
    