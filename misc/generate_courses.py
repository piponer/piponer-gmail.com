# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:08:22 2019

@author: 63184
"""

import json
import os
from matplotlib.pylab import *

def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, int64): return int(o)
    raise TypeError

def save_for_evaluation(path_save, results, dset_name):
    path_save_file = os.path.join(path_save, f'results_{dset_name}.json')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)
            
def single_course(code, title, url, lecturer, admin):
    cur_dict = dict()
    cur_dict['course_code'] = code
    cur_dict['course_title'] = title
    cur_dict['course_url'] = url
    cur_dict['lecturer'] = lecturer
    cur_dict['admin'] = admin
    cur_dict['credit'] = 6
    cur_dict['faculty'] = 'Faculty of Engineering'
    cur_dict['school'] = 'School of Computer Science and Engineering'
    cur_dict['campus'] = 'Kensington'
    cur_dict['hours'] = 48
    cur_dict['course_fee'] = 5970
    return cur_dict

if __name__ == '__main__':
    results = []
    results.append(single_course("COMP1521", "Computer Systems Fundamentals", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP1521/?q=comp1521&ct=all", "Dr Andrew Taylor", "Jashank Jeremy"))
    results.append(single_course("COMP3331", "Computer Networks and Applications", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP3331/?q=COMP3331&ct=all", "Salil Kanhere", "Nadeem Ahmed"))
    results.append(single_course("COMP2521", "Data Structures and Algorithms", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp2521/?q=comp2521", "Ashesh Mahidadia", "Mei Cheng Whale"))
    results.append(single_course("COMP1511", "Introduction to Programming", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp1511/?q=comp1511", "Marc Chee", "Andrew Bennett"))
    results.append(single_course("COMP9321", "Data Services Engineering", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9321/?q=comp9321", "Morty Al-Banna", "Mohammadali Yaghoubzadehfard"))
    results.append(single_course("COMP9311", "Database Systems", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9311/?q=comp9311", "Wenjie Zhang", "Kai Wang"))
    results.append(single_course("COMPMP2511", "Object-Oriented Design & Programming", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp2511/?q=comp2511", "Ashesh Mahidadia", "Robert Clifton-Everest"))
    results.append(single_course("COMP9020", "Foundations of Computer Science", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9020/?q=comp9020", "Paul Hunter", "Paul Hunter"))
    results.append(single_course("COM9444", "Neural Networks and Deep Learning", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9444/?q=comp9444", "Alan Blair", "Alexander Long"))
    results.append(single_course("COMP3311", "Database Systems", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP3311/?q=comp3311&ct=all", "John Shepherd", "Hayden Smith"))
    results.append(single_course("COMP1531", "Software Engineering Fundamentals", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/comp1531/?q=comp1531", "Hayden Smith", "Robert Clifton-Everest"))
    results.append(single_course("COP9024", "Data Structures and Algorithms", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/COMP9024/?q=comp9024&ct=all", "Michael Thielscher", "Michael Schofield"))
    results.append(single_course("COMP9315", "DBMS Implementation", "https://www.handbook.unsw.edu.au/postgraduate/courses/2020/comp9315/?q=comp9315", "John Shepherd", "Hayden Smith"))
    results.append(single_course("COMP9313", "Big Data Management", "https://www.handbook.unsw.edu.au/undergraduate/courses/2020/COMP9313/?q=comp9313&ct=all", "Morty Al-Banna", "Maisie Badami"))
    save_for_evaluation('.', results, 'course1')