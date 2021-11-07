# -*- coding: utf-8 -*-
"""
Front end streamlit app.
Upload data.
Pick response.
Pick tree type.
Create Tree.
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
from io import BytesIO
from PIL import Image

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
SCREEN_WIDTH_PERC = 60

#assert MAX_CLASSES <= CAT_LIMIT

# =============================================================================
# FUNCTIONS
# =============================================================================
def _max_width_(SCREEN_WIDTH_PERC):
    '''
    Sets pixel width of the streamlit app
    https://discuss.streamlit.io/t/custom-render-widths/81/7
    '''
    max_width_str = f"max-width: {SCREEN_WIDTH_PERC}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True)
    
def main():
    '''
    The Streamlit app
    '''
    _max_width_(SCREEN_WIDTH_PERC)
    st.title('Create a Decision Tree!')
    
    #placeholder variables
    response = 'SELECT A COLUMN'
    
    #divide page veritcally
    col1, col2 = st.columns(2)
    
    #left side is data upload
    with col1:
        #File upload
        uploaded_file = st.file_uploader(label='Upload csv file',\
                                         type=['csv'],\
                                         help='Upload a csv file that is tabular data, 5MB limit',\
                                         )
    #right side is selection boxes
    with col2:
        if uploaded_file is not None:
            #load data as pandas df
            df = dtm.loadData(uploaded_file)
            
            #get column headings
            cols = dtm.getColumns(df)
            
            #check column limit
            note = dtm.checkColLimit(cols, COL_LIMIT)
            if note is not None:
                st.error(note)
                    
            
            else:
                #TEMP show df
                #st.write(df)
                
                #get user to pick a column as response variable
                response = st.selectbox(label='Pick your response variable',\
                                            options=cols + ['SELECT A COLUMN'],\
                                            index=len(cols))
                #TEMP    
                #st.write('Response Variable:', response)
            
            #pick problem type
            if response != 'SELECT A COLUMN':
                tree_type = st.selectbox(label='Pick Tree type. (Regression or Classification)',\
                                         options=['Regression', 'Classification', 'PICK TREE TYPE'],\
                                         index=2)
                    
                #TEMP
                #st.write('Tree Type:', tree_type)
            
        #reveal "Go" Button (underneath columns
        if response != 'SELECT A COLUMN'\
        and tree_type != 'PICK TREE TYPE':
            go_button = st.button(label='Create Decision Tree')
            
            #Gooo!
            if go_button:
                #make response col dtype string if Classification
                df = dtm.makeRespStr(df, response, tree_type)
                
                #get class names if Classification
                class_names = dtm.getClassNames(df, response, MAX_CLASSES, tree_type)
                
                #check to see if there weren't too many classes
                if type(class_names) is str:
                    st.error(class_names)
                          
                else:
                    #determine categorical and numerical columns
                    dtype_dict = dtm.catOrNum(df, NUMERICS)
                    print('\n', dtype_dict, '\n')
                    
                    #process null values
                    df = dtm.processNulls(df, dtype_dict)
                    
                    #make column numeric if most values are
                    df, dtype_dict = dtm.makeNumeric(df, dtype_dict, MAKE_NUM_THRESH)
                    
                    #if too many values in reg response are categorical then terminate and warn
                    note = dtm.checkRegResponse(df, response, dtype_dict, tree_type)
                    if note is not None:
                        st.error(note)
                        
                    else:
                        #drop categorical columns if there are too many unique values
                        df, dtype_dict, dropped = dtm.dropUniques(df, dtype_dict, UNIQUE_THRESH)
                        if dropped != []:
                            st.warning('The following columns have been dropped as they contained too'+\
                                       f' many unique categories: \n{dropped}'+\
                                       '\nWere they meant to be categorical?\n')
                        
                        #limit number of categories in cat columns
                        df = dtm.limitCats(df, dtype_dict, CAT_LIMIT)
                        
                        #if Classification then ordinal encode the response (for sklearn)
                        df, dtype_dict = dtm.ordinalResponse(df, response, class_names, dtype_dict, tree_type)
                        
                        #one hot encode the categorical columns
                        df = dtm.dummyVars(df, dtype_dict)
                        
                        #train a tree
                        dtree = dtm.trainTree(df, tree_type, response)
        
                        #generate the tree graphic to BytesIO
                        mem_fig = dtm.genTree(df, dtree, class_names, response,\
                                              w=14, h=6, dpi=300, fontsize=8)
                        
                        #display as image on app
                        st.image(mem_fig)
                

# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
    main()