# -*- coding: utf-8 -*-
"""
workshop to edit the dtree strings
"""
# =============================================================================
# Imports
# =============================================================================
import re
import pydot 
from io import BytesIO, StringIO
from PIL import Image

# =============================================================================
# CONSTANTS
# =============================================================================
SEP = '$!@'
#PROB = 'Regression'
PROB = 'Classification'
#class_names = [1,1] #so len = 2
class_names = [1,1,1]

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

# =============================================================================
# EXECUTE
# =============================================================================
#load example
with open(r'C:\Users\JF\Desktop\git_projects\display_dtree_webapp\Data\example iris str.txt') as f:
    d_str = f.read()
    
    
#do regex
pattern = re.compile(r'label=(.*)\\nsamples')

#loop it in
exps = pattern.findall(d_str)

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
    

#true false arrows
# NO=============================================================================
# ##only do this on full trees
# if '12 -> 14' in d_str:
#     print(True)
# =============================================================================

pat_arrow = re.compile(r'\d+ -> \d+')
exps = pat_arrow.findall(d_str)

for exp in exps:
    to_and_from = exp.split(' -> ')
    
    if int(to_and_from[1]) - int(to_and_from[0]) == 1:
        d_str = d_str.replace(exp, exp + ' [labeldistance=2.5, labelangle=45, headlabel="True"]')
        
    else:
        d_str = d_str.replace(exp, exp + ' [labeldistance=2.5, labelangle=-45, headlabel="False"]')

#T/F = Y/N
d_str = d_str.replace('headlabel="True"', 'headlabel="Yes"')
d_str = d_str.replace('headlabel="False"', 'headlabel="No"')

#create png and save to memory
graph = pydot.graph_from_dot_data(d_str)[0]
#graph.set_dpi(dpi)
mem_fig_gv = BytesIO(graph.create_png())
image_plot = Image.open(mem_fig_gv)
image_plot

#hex to rgb
# =============================================================================
# ##Regression: Orange -> Green
# if PROB == 'Regression':
#     #find hex colours in string
#     colour_pat = re.compile(r'fillcolor=\"(#.*)\"')
#     colour_hexes = colour_pat.findall(d_str)
#     
#     #Loop
#     for hexy in colour_hexes:    
#         #Convert colour to RGB
#         rgb = hex_to_rgb(hexy)
#         
#         #Make green
#         red, green, blue = rgb
#         red = blue
#         rgb_new = (red, green, blue)
#         
#         #Convert back to hex
#         hex_new = rgb_to_hex(rgb_new)
#         
#         #replace string
#         d_str = d_str.replace(hexy, hex_new)
# 
# ##Binary Classification
# if PROB == 'Classification' and len(class_names) == 2:
#     #find hex colours in string
#     colour_pat = re.compile(r'fillcolor=\"(#.*)\"')
#     colour_hexes = colour_pat.findall(d_str)
#     
#     #Loop
#     for hexy in colour_hexes:    
#         #Convert colour to RGB
#         rgb = hex_to_rgb(hexy)
#         red, green, blue = rgb
#         
#         #check: orange or blue?
#         ##blue
#         if blue > red:
#             #make green
#             blue = red
#             
#         ##orange
#         else:
#             #make red
#             green = int(green * (green/255))
#             blue = green
#             red = 255
#             
#         rgb_new = (red, green, blue)
#         
#         #Convert back to hex
#         hex_new = rgb_to_hex(rgb_new)
#         
#         #replace string
#         d_str = d_str.replace(hexy, hex_new)
# 
# =============================================================================
#refactor
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


        
        
#create png and save to memory
graph = pydot.graph_from_dot_data(d_str)[0]
#graph.set_dpi(dpi)
mem_fig_gv = BytesIO(graph.create_png())
image_plot = Image.open(mem_fig_gv)
image_plot