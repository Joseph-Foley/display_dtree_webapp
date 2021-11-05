# -*- coding: utf-8 -*-
"""
Front end streamlit app
"""
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import streamlit as st

import dtree_modelling as dtm

#TEMP
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS
# =============================================================================
TREE_DEPTH = 3
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
UNIQUE_THRESH = 0.2
MAKE_NUM_THRESH = 0.95
MAX_CLASSES = 10
CAT_LIMIT = 10
COL_LIMIT = 100

#assert MAX_CLASSES <= CAT_LIMIT

# =============================================================================
# FUNCTIONS
# =============================================================================
def main():
    '''
    The Streamlit app
    '''
    st.write('Create a Decision Tree!')
    
    uploaded_file = st.file_uploader(label='Upload csv file',\
                                     type=['csv'],\
                                     help='Upload a csv file that is tabular data, 5MB limit',\
                                     )
        
    if uploaded_file is not None:
        #load data as pandas df
        df = dtm.loadData(uploaded_file)
        
        #TEMP show df
        st.write(df)
        
        #get column headings
        cols = dtm.getColumns(df)
        
        #get user to pick a column as response variable
        response = st.selectbox(label='Pick your response variable',\
                                    options=cols + ['SELECT A COLUMN'],\
                                    index=len(cols))
        #TEMP    
        st.write('Response Variable:', response)
        
        #pick problem type
        if response != 'SELECT A COLUMN':
            tree_type = st.selectbox(label='Pick Tree type. (regression or classification)',\
                                     options=['regression', 'classification', 'PICK TREE TYPE'],\
                                     index=2)
                
            #TEMP
            st.write('Tree Type:', tree_type)
        
        #reveal "Go" Button
        if response != 'SELECT A COLUMN'\
        and tree_type != 'PICK TREE TYPE':
            go_button = st.button(label='Create Decision Tree')
            
            #Gooo!
            if go_button:
                st.write('gooo!\n\n\n')
                
                
                #make response col dtype string if classification
                df = dtm.makeRespStr(df, response, tree_type)
                
                #get class names if classification
                class_names = dtm.getClassNames(df, response, MAX_CLASSES, tree_type)
                
                #determine categorical and numerical columns
                dtype_dict = dtm.catOrNum(df, NUMERICS)
                print('\n', dtype_dict, '\n')
                
                #process null values
                df = dtm.processNulls(df, dtype_dict, response)
                
                #make column numeric if most values are
                df, dtype_dict = dtm.makeNumeric(df, dtype_dict, MAKE_NUM_THRESH)
                
                #if too many values in reg response are categorical then terminate and warn
                dtm.checkRegResponse(df, response, dtype_dict, tree_type)
                
                #drop categorical columns if there are too many unique values
                df, dtype_dict = dtm.dropUniques(df, dtype_dict, UNIQUE_THRESH)
                
                #limit number of categories in cat columns
                df = dtm.limitCats(df, dtype_dict, CAT_LIMIT)
                
                #if classification then ordinal encode the response (for sklearn)
                df, dtype_dict = dtm.ordinalResponse(df, class_names, dtype_dict, tree_type)
                
                #one hot encode the categorical columns
                df = dtm.dummyVars(df, dtype_dict)
                
                #train a tree
                dtree = dtm.trainTree(df, tree_type, response)
                
                #generate the tree graphic
                #st.pyplot(dtm.genTree(df, dtree, class_names, response))
                fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,12))
                
                plot_tree(dtree, feature_names=df.drop(response, axis=1).columns,\
                class_names=class_names, filled=True, rounded=True, precision=2,\
                proportion=True, impurity=False)
                
                st.pyplot(fig)

# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
    main()