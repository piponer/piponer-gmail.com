# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:36:58 2019

@author: 63184
"""

import json
from raw2dict import save_obj_dict

def fix_idx(path, w2idx_path):
    w_dict = dict()
    cnt = 0
    with open(path) as f:
        w_dict = json.load(f)
    res_dict = dict()
    for key in w_dict:
        res_dict[key] = cnt
        cnt += 1
    save_obj_dict(w2idx_path, res_dict)

if __name__ == '__main__':
    fix_idx('word_md/sub_words.json', 'word_md/word2idx.json')