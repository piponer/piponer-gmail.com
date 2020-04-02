# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:26:35 2019

@author: 63184
"""

import json
from utils import *

def analyse(fname, subw):
    w2sw = generate_word2subword(subw)
    max_context = 0
    max_single_context = 0
    max_response = 0
    context_len = 0
    context_len_m = 20
    min_q = 10000
    min_r = 10000
    with open(fname, encoding='utf-8') as f:
        for line in f:
            current = json.loads(line.strip())
            context = current['context']
            response = current['response']
            context_split = tokenize_new(context, w2sw)
            response_split = tokenize_new(response, w2sw)
            context_next = context.split(' </s> ')
            if len(context_next) > context_len:
                context_len = len(context_next)
            if len(context_next) < context_len_m:
                context_len_m = len(context_next)
            context_next_split = [tokenize_new(e, w2sw) for e in context_next]
            if len(context_next_split[-1]) > max_single_context:
                max_single_context = len(context_next_split[-1])
            if len(context_next_split[-1]) < min_q:
                min_q = len(context_next_split[-1])
            if len(response_split) < min_r:
                min_r = len(response_split)
            if len(context_split) > max_context:
                max_context = len(context_split)
            if len(response_split) > max_response:
                max_response = len(response_split)
    print('max_context:', max_context, '; max_single_context:', max_single_context, '; max_response:', max_response, '; context_len:', context_len, '; context_len_m:', context_len_m, '; min_q:', min_q, '; min_r:', min_r)
    
if __name__ == '__main__':
    analyse('useful_dataset/train.json', 'word_md/vocab.json')