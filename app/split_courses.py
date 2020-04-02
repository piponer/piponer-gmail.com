# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:09:33 2019

@author: 63184
"""

import json

with open('course.json') as f:
    c_dict = json.load(f)

stage = [17, 33]

stage_id = 0

fname = ['course1.json', 'course2.json', 'course3.json']

cur_dict = dict()

for i, (k, v) in enumerate(c_dict.items()):
    if i in stage:
        jsObj = json.dumps(cur_dict)
        fileObj = open(fname[stage_id], 'w')
        fileObj.write(jsObj)
        fileObj.close()
        cur_dict = dict()
        stage_id += 1
    course_info = k.split()
    c_id = course_info[0]
    c_name = ' '.join(course_info[2:])
    cur_dict[c_id] = c_name

jsObj = json.dumps(cur_dict)
fileObj = open(fname[stage_id], 'w')
fileObj.write(jsObj)
fileObj.close()