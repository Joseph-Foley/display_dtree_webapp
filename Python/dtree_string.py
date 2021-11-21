# -*- coding: utf-8 -*-
"""
workshop to edit the dtree strings
"""
import re
import pydot 
from io import BytesIO, StringIO
from PIL import Image

SEP = '$!@'


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