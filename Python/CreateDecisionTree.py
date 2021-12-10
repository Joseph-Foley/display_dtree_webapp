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
from PIL import Image

import dtree_modelling as dtm

#TEMP
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from io import BytesIO

import os
import shutil

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
SEP = '$!@'
RANDOM_STATE = 99

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

def editIndexFile():
    """
    Take index.html file in repo and move it to base streamlit folder
    """
    #get streamlit file location
    st_dir = os.path.join(os.path.dirname(st.__file__), 'static')
    
    #replace the index file there with one in current dir
    shutil.copyfile('index.html', os.path.join(st_dir, 'index.html'))
    
    #replace favicon image too
    shutil.copyfile('favicon.png', os.path.join(st_dir, 'favicon.png'))
        
def main():
    '''
    The Streamlit app
    '''
    #replace default html template
    editIndexFile()
    
    #set desired width
    _max_width_(SCREEN_WIDTH_PERC)
    
    #Title
    st.title('Create a Decision Tree!')
    
    #Description
    st.markdown('__Got some data? Want Machine Learning to map it out as a Decision Tree?__')
    st.markdown('__Then drag and drop it right here!__')
    
    #side bar pop up tutorial
    how_work = st.button('How does this work?')
    if how_work:
        st.sidebar.subheader('How does this work?')
        
        st.sidebar.write('Make sure your data is in csv file format.')
        st.sidebar.write('Make sure your data is in the form of a table with the first row (and only the first) being your column headers.')
        st.sidebar.write('*It helps to remove columns that you don’t think are relevant such as id columns.')
        image = Image.open(r'../Images/iris_table_format.PNG')
        st.sidebar.image(image, use_column_width=True)
        
        st.sidebar.write('Drag and drop your csv file into the upload box or find your data by selecting the “browse files” button.')
        image = Image.open(r'../Images/dragNdrop.PNG')
        st.sidebar.image(image, use_column_width=True)
        
        st.sidebar.write('Select the column for the variable you are trying to predict.')
        st.sidebar.write('Select the Decision Tree type you require.')
        st.sidebar.write('*Choose “Classification” if you are try to predict the class or type of something.')
        st.sidebar.write('*Choose “Regression” if you are trying to predict some kind of quantity or continuous value.')
        st.sidebar.write('Select the “Create Decision Tree” button to generate your tree.')
        image = Image.open(r'../Images/iris_tree.png')
        st.sidebar.image(image, use_column_width=True)
        
        st.sidebar.markdown('Click the icon at the top right of the Tree to enlarge the image')
        image = Image.open(r'../Images/enlarge_icon.PNG')
        st.sidebar.image(image)
        
        st.sidebar.markdown('__Interpreting Nodes__')
        st.sidebar.markdown('__Classification:__')
        st.sidebar.markdown('‘samples’ shows the percentage of your data that is represented by the node')
        st.sidebar.markdown('‘value’ shows the proportion of each class at the node.')
        st.sidebar.markdown('‘class’ the class that is most represented by the node.')
        st.sidebar.markdown('The color is an indication of how sure the tree is of its class prediction. The darker the color, the more sure it is.')
        image = Image.open(r'../Images/class_nodes.PNG')
        st.sidebar.image(image, use_column_width=True)
        
        st.sidebar.markdown('__Regression:__')
        st.sidebar.markdown('‘samples’ shows the percentage of your data that is represented by the node')
        st.sidebar.markdown('‘value’ shows the average value of your response variable at the node.')
        st.sidebar.markdown('The color reflects the magnitude of the value. The larger the value, the darker the color.')
        image = Image.open(r'../Images/reg_nodes.PNG')
        st.sidebar.image(image, use_column_width=True)
        
    #Disclaimer
    st.write('DISCLAIMER: We do not collect any of the data you use on this site! Cloud storage is expensive and we’d rather not pay for it!')
    
    #placeholder variables
    response = 'SELECT A COLUMN'
    go_button = False
    
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
                df = dtm.dummyVars(df, dtype_dict, SEP)
                
                #train a tree
                dtree = dtm.trainTree(df, tree_type, response, RANDOM_STATE)

# =============================================================================
#                 #generate the tree graphic to BytesIO
#                 mem_fig = dtm.genTree(df, dtree, class_names, response, tree_type,\
#                                       w=14, h=6, dpi=300, fontsize=8)
#                 
#                 #display as image on app
#                 st.image(mem_fig)
# =============================================================================
                
                #generate the tree graphic to BytesIO
                fig = dtm.genTreeGV(df, dtree, class_names, response, tree_type, SEP)
                
                #display as image on app
                st.pyplot(fig)
                                    
                                    
# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
    main()