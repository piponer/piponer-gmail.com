# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:50:37 2019

@author: 63184
"""

import re, collections, json

def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            line_dict = json.loads(line.strip())
            words = line_dict['context'].lower().split() + line_dict['response'].lower().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_vocab2(filename):
    vocab = collections.defaultdict(int)
    myDict = dict()
    with open(filename) as f:
        myDict = json.load(f)
    for k, v in myDict.items():
        words = v['message_body'].lower().strip().split() + v['course'].lower().split()
        for word in words:
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens

def save_obj_dict(path, myDict):
    jsObj = json.dumps(myDict)
    fileObj = open(path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print('data have been saved to', path)
    
def bpe(srcname, dstname, dstname_vo, target_num):
    vocab = get_vocab(srcname)
    #vocab = get_vocab2(srcname)
    pre_sub_words = get_tokens(vocab)
    cnt = 0
    while len(pre_sub_words) < target_num:
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        store_best = pairs[best]
        vocab = merge_vocab(best, vocab)
        pre_sub_words = get_tokens(vocab)
        cnt += 1
        if cnt % 1 == 0:
            print(cnt, 'iteration finish, the size of sub words is:', len(pre_sub_words), ', best freq is:', store_best)
    print('there are', len(pre_sub_words), 'sub words.')
    save_obj_dict(dstname, pre_sub_words)
    save_obj_dict(dstname_vo, vocab)
    
if __name__ == '__main__':
    bpe('useful_dataset/train.json', 'sub_words.json', 'vocab.json', 32000)