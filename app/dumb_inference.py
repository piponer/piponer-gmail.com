# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 02:43:09 2019

@author: 63184
"""

import math
from .botnet import *
import os, argparse

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import *
from .botnet import Botnet, Botnet_loss, logit2pred_unique, logit2pred, belu, raw2gold_pred, raw2gold_truth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
# 设置logging,同时输出到文件和屏幕
import logging

logger = logging.getLogger()  # 不加名称设置root logger
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# 使用FileHandler输出到文件
if not os.path.exists('log'):
    os.makedirs('log')
fh = logging.FileHandler('log/log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)
# logger.info('this is info message')
################################################################

def construct_hyper_param(parser):
    #general
    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=4, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--shuffle_train',
                        default=False,
                        action='store_true',
                        help="If present, training data will shuffle.")
    parser.add_argument('--trained', default=False, action='store_true', help='start from checkpoint')
    
    #meb
    parser.add_argument('--emb_dim', default=128, type=int)

    #path Parameters
    parser.add_argument("--subword",
                        default='app/word_md/vocab.json', type=str,
                        help="word to subword.")
    parser.add_argument("--word_dict",
                        default='app/word_md/word2idx.json', type=str,
                        help="The vocabulary of all words.")
    parser.add_argument("--symbol_dict",
                        default='app/sig.json', type=str,
                        help="All of the symbols.")
    parser.add_argument("--train_data",
                        default='app/useful_dataset/train.json', type=str,
                        help="The training data.")
    parser.add_argument("--dev_data",
                        default='app/useful_dataset/val.json', type=str,
                        help="The training data.")
    parser.add_argument("--data_dict",
                        default='app/id2content.json', type=str,
                        help="The original data dict.")
    parser.add_argument("--model_path",
                        default='app/model_saved/model_botnet_best.pt', type=str,
                        help="The best belu score model.")
    parser.add_argument("--model_path_loss",
                        default='app/model_saved/model_botnet_best_loss.pt', type=str,
                        help="The best loss model.")

    #Botnet module parameters
    parser.add_argument('--lr', default=1e-3, type=float, help="learning rate.")
    parser.add_argument('--tf_emb_size', default=256, type=int, help="transformer hidden size.")
    parser.add_argument('--lS_lstm', default=2, type=int, help="knowledge lstm layers number.")
    parser.add_argument('--dr', default=0.1, type=float, help="dropout rate.")
    parser.add_argument("--nb_tf_bo", default=12, type=int, help="The number of transformer in transformer body.")
    parser.add_argument("--hidden_tf_bo", default=512, type=int, help="The hidden size of feed forward layer in transformer body.")
    parser.add_argument("--heads", default=8, type=int, help="The number of heads in multi-header attention layer.")
    parser.add_argument("--max_len", default=512, type=int, help="The number of tokens in a sequence.")

    args = parser.parse_args()

    return args

def get_model_botnet(args, w2id, id2w, w2sw, trained=False, path_model=None):
    model = Botnet(w2id, id2w, w2sw, args.tf_emb_size, args.lS_lstm, args.dr, args.nb_tf_bo, args.hidden_tf_bo, args.heads, args.max_len, args.emb_dim)
    model = model.to(device)
    
    if trained:
        assert path_model != None
        
        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model_botnet'])
        
    return model

def inference(inputs, word_set, sig, data_dict, n=4):
    best_score = 0.0
    best_target = "I don't know."
    input_tokens = tokenize(inputs, word_set, sig)
    for _, v in data_dict.items():
        cur_mesg = v['message_body']
        cur_mesg_tokens = tokenize(cur_mesg, word_set, sig)
        if len(cur_mesg_tokens) < 5 or 'children_id' not in v:
            continue
        cur_target = data_dict[str(v['children_id'])]['message_body']
        cur_target_tokens = tokenize(cur_target, word_set, sig)
        if len(cur_target_tokens) < 3:
            continue
        n_gram_score_ave = 0
        for i in range(1, n + 1):
            n_gram_score_ave += math.log(n_gram_acc(input_tokens, cur_mesg_tokens, i))
        n_gram_score_ave /= n
        cur_belu_score = BP(input_tokens, cur_mesg_tokens) * math.exp(n_gram_score_ave)
        if cur_belu_score > best_score:
            best_score = cur_belu_score
            best_target = cur_target
    return best_target, best_score

def initial():
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)
    w2id, id2w = get_w2id_id2w(args.word_dict)
    w2sw = generate_word2subword(args.subword)
    model = get_model_botnet(args, w2id, id2w, w2sw, trained=True, path_model=args.model_path)
    
    word_set = get_word_set('app/words.json')
    sig = get_sig('app/sig.json')
    data_dict = load_dict('app/id2content.json')
    return word_set, sig, data_dict, args, w2id, id2w, w2sw, model

def generate_single_input(cur_question, cur_answer, word2sub, max_part=224, max_len=512, pre_question=None, pre_answer=None, old_kl=None):
    l_hs = []
    nlu_hpu = []
    seq_input = []
    first_sep_encoder = []
    first_sep_decoder = []
    cls_idx = []
    seg_idx = []
    max_input_len = max_len - 1
    max_seg_len = max_len
    if pre_question == None:
        pre_context = ["Nothing."]
    else:
        pre_context = [pre_question, pre_answer]
    now_context = cur_question
    response = cur_answer
    l_hs.append(len(pre_context))
    nlu_hpu += [tokenize_new(e, word2sub) for e in pre_context]
    first_part = get_part_new(now_context, word2sub, max_part)
    if response:
        second_part = response
    else:
        second_part = []
    seq_input1 = []
    seg_idx1 = []
    seg_idx1.append(1)# for knowledge
    seq_input1.append('[sep]')
    seg_idx1.append(1)
    seq_input1 += first_part
    seg_idx1.extend([2 for _ in range(len(first_part))])
    seq_input1.append('[sep]')
    seg_idx1.append(2)
    first_sep_decoder.append(len(seq_input1))
    if second_part:
        seq_input1 += second_part
        seg_idx1.extend([3 for _ in range(len(second_part))])
    seq_input1.append('[sep]')
    seg_idx1.append(3)
    first_sep_encoder.append(len(seq_input1))
    seq_input1.append('[cls]')
    seg_idx1.append(3)
    cls_idx.append(len(seq_input1))
    seq_input1 += ['[pad]' for _ in range(max_input_len - len(seq_input1))]
    seg_idx1 += [0 for _ in range(max_seg_len - len(seg_idx1))]
    seq_input.append(seq_input1)
    seg_idx.append(seg_idx1)
    
    return l_hs, nlu_hpu, seq_input, seg_idx, cls_idx, first_sep_encoder, first_sep_decoder, old_kl

def test_botnet(args, w2id, id2w, w2sw, single_input, model, pre_question=None, pre_answer=None, old_kl=None, iter_nb=30):
    model.eval()
    cur_answer = None
    for i in range(iter_nb):
        l_hs, nlu_hpu, seq_input, seg_idx, cls_idx, first_sep_encoder, first_sep_decoder, kl = generate_single_input(single_input, cur_answer, w2sw, pre_question=pre_question, pre_answer=pre_answer, old_kl=old_kl)
        mode = 1 if pre_question == None else 2
        logit, label, kl = model(seq_input, [], seg_idx, nlu_hpu, l_hs, cls_idx, first_sep_encoder, first_sep_decoder, mode=mode, old_kl=kl)
        y_pred = logit2pred_unique(logit)
        gold_pred = raw2gold_pred(y_pred, cls_idx, first_sep_decoder)
    
        cur_answer = [id2w[w_id] for w_id in gold_pred[0]]
    
        print('example prediction:', cur_answer)
        
        if cur_answer[-1] == '[eof]':
            break
    
    cur_answer = ''.join(cur_answer).replace('</w>', ' ').replace('[eof]', '').strip()
    #print(kl)
    return cur_answer, kl

def main_inference(inputs, args, w2id, id2w, w2sw, model, word_set, sig, data_dict, pre_question=None, pre_answer=None, old_kl=None):
    answer1, score1 = inference(inputs, word_set, sig, data_dict)
    print(answer1)
    if score1 > 0.1:
        return {'a': answer1, 'kl': torch.zeros(3,3), 'label': False}
    else:
        answer1, kl = test_botnet(args, w2id, id2w, w2sw, inputs, model, pre_question=pre_question, pre_answer=pre_answer, old_kl=old_kl)
        return {'a': answer1, 'kl': kl, 'label': True}

def core_inference(text):
    word_set, sig, data_dict, args, w2id, id2w, w2sw, model = initial()
    Q1 = text
    out_dict = main_inference(Q1, args, w2id, id2w, w2sw, model, word_set, sig, data_dict)
    return out_dict

if __name__ == '__main__':
    word_set, sig, data_dict, args, w2id, id2w, w2sw, model = initial()
    Q1 = 'How to build a java project?'
    A1, kl = main_inference(Q1, args, w2id, id2w, w2sw, model, word_set, sig, data_dict)
    print('user:', Q1)
    print('bot:', A1)
    Q2 = 'Why you want to do that?'
    A2, kl = main_inference(Q2, args, w2id, id2w, w2sw, model, word_set, sig, data_dict, Q1, A1, kl)
    print('user:', Q2)
    print('bot:', A2)