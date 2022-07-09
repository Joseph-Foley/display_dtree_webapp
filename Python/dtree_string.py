# -*- coding: utf-8 -*-
"""
Utility script that edits the strings which make up the dtree.
Changes from default:
    Node colours
    Categorical node operators
    Arrow text
"""
# =============================================================================
# Imports
# =============================================================================
import re
#import pydot
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
# =============================================================================
# CONSTANTS
# =============================================================================
SEP = '$!@'
PROB = 'Regression'
#PROB = 'Classification'
class_names = [1,1] #so len = 2
#class_names = [1,1,1]

if PROB == 'Regression':
    class_names = None

# =============================================================================
# FUNCTIONS
# =============================================================================
def hex_to_rgb(value: str):
    '''
    https://stackoverflow.com/questions/29643352
    '''
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb: tuple):
    '''
    https://stackoverflow.com/questions/29643352
    '''
    return '#%02x%02x%02x' % rgb

def changeCatNodes(d_str, SEP):
    """
    Categorical nodes to be : Feature = Category
    """
    #regex expression to get feature nodes
    pattern = re.compile(r'label=(.*)\\nsamples')
    
    #find em
    exps = pattern.findall(d_str)
    
    #replace em
    for exp in exps:
        if SEP in exp:
            #get left and right
            components = exp.split(SEP)
            left_side = components[0]
            right_side = components[1].split(' <=')[0]
            
            #combine
            new_str = ' = '.join([left_side, right_side])
            
            #insert into main string
            d_str = d_str.replace(exp, new_str)
    
    return d_str

def changeArrowText(d_str):
    '''
    Arrow text to be on EVERY arrow. Changed to "Yes" & "No"
    '''
    #find arrow strings
    pat_arrow = re.compile(r'\d+ -> \d+')
    exps = pat_arrow.findall(d_str)
    
    #Add text to all arrows
    for exp in exps:
        to_and_from = exp.split(' -> ')
        
        if int(to_and_from[1]) - int(to_and_from[0]) == 1:
            str_T = ' [labeldistance=2.5, labelangle=45, headlabel="True"]'
            d_str = d_str.replace(exp, exp + str_T, 1)
            
        else:
            str_F = ' [labeldistance=2.5, labelangle=-45, headlabel="False"]'
            d_str = d_str.replace(exp, exp + str_F, 1)
    
    #T/F = Y/N
    d_str = d_str.replace('headlabel="True"', 'headlabel="Yes"')
    d_str = d_str.replace('headlabel="False"', 'headlabel="No"')
    
    return d_str

def changeHexColours(d_str, class_names):
    '''
    Change default d_tree colours for regression and binary classification
    to red and green. Other models, no change.
    '''
    #check if reg or bin classification
    if class_names is None \
    or len(class_names) == 2:
        #find hex colours in string
        colour_pat = re.compile(r'fillcolor=\"(#.*)\"')
        colour_hexes = colour_pat.findall(d_str)   
        
        #Loop
        for hexy in colour_hexes:    
            #Convert colour to RGB
            rgb = hex_to_rgb(hexy)
            red, green, blue = rgb
            
            #(Reg) orange -> green
            if class_names is None:
                #make green
                red = blue
                
            #(Class) blue -> green
            elif blue > red:
                #make green
                blue = red
                
            #(Class) orange -> red
            elif red > blue:
                #make red
                green = int(green * (green/255))
                blue = green
                red = 255
            
            #new colour
            rgb_new = (red, green, blue)
            
            #Convert back to hex
            hex_new = rgb_to_hex(rgb_new)
            
            #replace string
            d_str = d_str.replace(hexy, hex_new)
            
    return d_str

def changeDtreeString(d_str, SEP, class_names):
    """
    Combines the 3 string changing functions
    """
    d_str = changeCatNodes(d_str, SEP)
    d_str = changeArrowText(d_str)
    d_str = changeHexColours(d_str, class_names)
    
    return d_str

# =============================================================================
# EXECUTE
# =============================================================================
if __name__ =='__main__':
    #load example
    with open(r'C:\Users\JF\Desktop\git_projects\display_dtree_webapp\Data\example US_pop str.txt') as f:
        d_str = f.read()
    
    #change the string
    d_str = changeDtreeString(d_str, SEP, class_names)  
         
    #create png and save to memory
    graph = pydot.graph_from_dot_data(d_str)[0]
    #graph.set_dpi(dpi)
    mem_fig_gv = BytesIO(graph.create_png())
    image_plot = Image.open(mem_fig_gv)
    image_plot
    
    

    dpi=100
    fig, axes = plt.subplots(nrows = 1,ncols = 1, dpi=dpi)
    axes.title.set_text('{PROB} Decision Tree for {response}')
    plt.axis('off')
    imgplot = plt.imshow(image_plot)
    plt.show()