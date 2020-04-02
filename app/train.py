# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:23:49 2019

@author: 63184
"""

import os, argparse

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from botnet import Botnet, Botnet_loss, logit2pred, belu, raw2gold_pred, raw2gold_truth

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
                        default='word_md/vocab.json', type=str,
                        help="word to subword.")
    parser.add_argument("--word_dict",
                        default='word_md/word2idx.json', type=str,
                        help="The vocabulary of all words.")
    parser.add_argument("--symbol_dict",
                        default='sig.json', type=str,
                        help="All of the symbols.")
    parser.add_argument("--train_data",
                        default='useful_dataset/train.json', type=str,
                        help="The training data.")
    parser.add_argument("--dev_data",
                        default='useful_dataset/val.json', type=str,
                        help="The training data.")
    parser.add_argument("--data_dict",
                        default='id2content.json', type=str,
                        help="The original data dict.")
    parser.add_argument("--model_path",
                        default='model_saved/model_botnet_best.pt', type=str,
                        help="The best belu score model.")
    parser.add_argument("--model_path_loss",
                        default='model_saved/model_botnet_best_loss.pt', type=str,
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

def get_opt_botnet(args, model):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
    
    return opt

def train_botnet(args, w2id, id2w, w2sw, train_loader, model, opt, st_pos=0):
    model.train()
    
    ave_loss = 0
    ave_belu = 0
    cnt = 0
    nb_batch = 0
    
    for iB, t in enumerate(train_loader):
        nb_batch += 1
        cnt += len(t)
        if cnt < st_pos:
            continue
        l_hs, nlu_hpu, seq_input, seq_target, seg_idx, cls_idx, first_sep_encoder, first_sep_decoder = generate_inputs_new(t, w2sw)
        logit, label, kl = model(seq_input, seq_target, seg_idx, nlu_hpu, l_hs, cls_idx, first_sep_encoder, first_sep_decoder)
        loss = Botnet_loss(logit, label, cls_idx, first_sep_decoder)
        #print(loss)
        loss = loss.sum()
        y_pred = logit2pred(logit)
        #print(y_pred.size())
        #print(label.size())
        gold_pred = raw2gold_pred(y_pred, cls_idx, first_sep_decoder)
        gold_truth = raw2gold_truth(label, cls_idx, first_sep_decoder)
        belu_score = sum(belu(gold_pred, gold_truth))
        
        #print(belu_score)
        
        #print(loss.item())
        
        if iB % args.accumulate_gradients == 0: # mode
            # at start, perform zero_grad
            opt.zero_grad()
            loss.backward()
            if args.accumulate_gradients == 1:
                opt.step()
        elif iB % args.accumulate_gradients == (args.accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()
            
        ave_loss += loss.item()
        ave_belu += belu_score
        
        #print(ave_loss)
        #print(ave_belu)
        #print(len(id2w))
        #print(gold_truth)
        
        if nb_batch % 1 == 0:
            logger.info('%d - th train data batch -> loss: %.4f; belu: %.4f' % 
                (nb_batch, ave_loss / cnt, ave_belu / cnt))
            tmp = ''.join([id2w[w_id] for w_id in gold_pred[0]]).replace('</w>', ' ')
            tmp = tmp.split('[eof]')[0]
            tmp = tmp.replace(' a i ', ' ai ').split()
            res = []
            for e in tmp:
                if not res or res[-1] != e:
                    res.append(e)
            res = ' '.join(res)
            print('example prediction:', res)
            print('example target:', ''.join([id2w[w_id] for w_id in gold_truth[0]]).replace('</w>', ' ').replace('[eof]', ''))
    
    ave_loss /= cnt
    ave_belu /= cnt
    
    metric = [ave_loss, ave_belu]
    
    return metric

def dev_botnet(args, w2id, id2w, w2sw, dev_loader, model, st_pos=0):
    model.eval()
    
    ave_loss = 0
    ave_belu = 0
    cnt = 0
    nb_batch = 0
    
    for iB, t in enumerate(dev_loader):
        nb_batch += 1
        cnt += len(t)
        if cnt < st_pos:
            continue
        l_hs, nlu_hpu, seq_input, seq_target, seg_idx, cls_idx, first_sep_encoder, first_sep_decoder = generate_inputs_new(t, w2sw)
        logit, label, kl = model(seq_input, seq_target, seg_idx, nlu_hpu, l_hs, cls_idx, first_sep_encoder, first_sep_decoder)
        loss = Botnet_loss(logit, label, cls_idx, first_sep_decoder, False)
        loss = loss.sum()
        y_pred = logit2pred(logit)
        gold_pred = raw2gold_pred(y_pred, cls_idx, first_sep_decoder)
        gold_truth = raw2gold_truth(label, cls_idx, first_sep_decoder)
        belu_score = sum(belu(gold_pred, gold_truth))
        
        ave_loss += loss.item()
        ave_belu += belu_score
        
        if nb_batch % 1000 == 0:
            logger.info('%d - th dev data batch -> loss: %.4f; belu: %.4f' % 
                (nb_batch, ave_loss / cnt, ave_belu / cnt))
            print('example prediction:', ' '.join([id2w[w_id] for w_id in gold_pred[0]]))
            print('example target:', ' '.join([id2w[w_id] for w_id in gold_truth[0]]))
            
    ave_loss /= cnt
    ave_belu /= cnt
    
    metric = [ave_loss, ave_belu]
    
    return metric

if __name__ == '__main__':
    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)
    
    #word_set = get_word_set(args.word_dict)
    #sig = get_sig(args.symbol_dict)
    #data_dict = load_dict(args.data_dict)
    
    train_loader, dev_loader = get_loader_new(args.train_data, args.dev_data, args.bS, shuffle_train=args.shuffle_train)
    
    w2id, id2w = get_w2id_id2w(args.word_dict)
    w2sw = generate_word2subword(args.subword)
    #for kv in w2sw.items():
    #    print(kv)
    '''
    word_emb, w2id, _ = load_emb(args.word_emb, w2id, args.emb_dim)
    
    id2w = dict()
    
    for w, w_id in w2id.items():
        id2w[w_id] = w
    '''
    
    #course2id = load_course(args.course)
    
    model = get_model_botnet(args, w2id, id2w, w2sw, trained=args.trained, path_model=args.model_path)
    
    opt = get_opt_botnet(args, model)
    
    loss_ave_best = float('inf')
    belu_ave_best = -1
    epoch_best_belu = -1
    epoch_best_loss = -1
    for epoch in range(args.tepoch):
        
        metric_train = train_botnet(args, w2id, id2w, w2sw, train_loader, model, opt)
        
        with torch.no_grad():
            metric_dev = dev_botnet(args, w2id, id2w, w2sw, dev_loader, model)
        
        loss_temp, belu_temp = metric_dev
        if belu_temp > belu_ave_best:
            state = {'model_botnet': model.state_dict()}
            torch.save(state, os.path.join(args.model_path))
            belu_ave_best = belu_temp
            epoch_best_belu = epoch
        print(f" Best dev ave belu: {belu_ave_best} at epoch: {epoch_best_belu}")
        if loss_temp < loss_ave_best:
            state = {'model_botnet': model.state_dict()}
            torch.save(state, os.path.join(args.model_path_loss))
            loss_ave_best = loss_temp
            epoch_best_loss = epoch
        print(f" Best dev ave loss: {loss_ave_best} at epoch: {epoch_best_loss}")