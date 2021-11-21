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
with open(r'C:\Users\JF\Desktop\git_projects\display_dtree_webapp\Data\example dtree str.txt') as f:
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
    

#create png and save to memory
graph = pydot.graph_from_dot_data(d_str)[0]
#graph.set_dpi(dpi)
mem_fig_gv = BytesIO(graph.create_png())
image_plot = Image.open(mem_fig_gv)
image_plot