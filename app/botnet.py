# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:44:18 2019

@author: 63184
"""

from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from .utils import *
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def logit2pred(logit):
    return torch.argmax(logit, -1)

def logit2pred_unique(logit, wordSize=64000):
    logit = logit.squeeze(0)
    order = torch.argsort(-logit, dim=-1)
    pred = torch.zeros(1, order.size()[0]).long()
    previous = -1
    for i in range(pred.size()[-1]):
        for idx in order[i]:
            if idx < wordSize and idx != previous or idx == wordSize + 3:
                pred[0][i] = idx
                previous = idx.item()
                break
    return pred

def raw2gold_pred(pred, cls_idx, sep_decoder):
    pred = pred.tolist()
    gold_pred = []
    for i, pred1 in enumerate(pred):
        ed = cls_idx[i] - 1
        st = sep_decoder[i]
        gold_pred.append(pred1[st: ed])
        #gold_pred.append(pred1[: cls_idx[i] + 1])
    return gold_pred

def raw2gold_truth(truth, cls_idx, sep_decoder):
    truth = truth.tolist()
    gold_truth = []
    for i, truth1 in enumerate(truth):
        ed = cls_idx[i] - 1
        st = sep_decoder[i]
        gold_truth.append(truth1[st: ed])
    return gold_truth

def n_gram_acc(pred, truth, n, elip=1e-13):
    #print('pred:', pred)
    #print('truth:', truth)
    if len(pred) < n and len(truth) < n:
        return 1.0 - elip
    elif len(pred) < n:
        return 0.0 + elip
    elif len(truth) < n:
        return 1.0 - elip
    else:
        pred_list = []
        for st in range(len(pred) - n + 1):
            pred_list.append(tuple(pred[st: st + n]))
        truth_list = []
        for st in range(len(truth) - n + 1):
            truth_list.append(tuple(truth[st: st + n]))
        cnt_pred = defaultdict(int)
        cnt_truth = defaultdict(int)
        for token in pred_list:
            cnt_pred[token] += 1
        for token in cnt_pred:
            cnt_truth[token] += truth_list.count(token)
        sum_pred = sum([v for _, v in cnt_pred.items()])
        sum_truth = sum([v for _, v in cnt_truth.items()])
        res = (sum_truth + elip) / (sum_pred + elip)
        #print('pred:', sum_pred)
        #print('truth', sum_truth)
        return res
    
def BP(pred, truth, elip=1e-13):
    if len(pred) >= len(truth):
        return 1.0 - elip
    else:
        return math.exp(1.0 - elip  - (len(truth) + elip) / (len(pred) + elip))

def belu(y_pred, y_truth, n=4):
    #print(y_pred)
    #print(y_truth)
    #y_pred = y_pred.tolist()
    #y_truth = y_truth.tolist()
    belu_score = []
    for ib, y_pred1 in enumerate(y_pred):
        y_truth1 = y_truth[ib]
        #print('y_pred1:', len(y_pred1))
        #print('y_truth1:', len(y_truth1))
        n_gram_score_ave = 0
        for i in range(1, n + 1):
            n_gram_score_ave += math.log(n_gram_acc(y_pred1, y_truth1, i))
        n_gram_score_ave /= n
        belu_score1 = BP(y_pred1, y_truth1) * math.exp(n_gram_score_ave)
        belu_score.append(belu_score1)
    return belu_score#[bs]    

def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def word2sig(word_size, word_id):
    if word_id < word_size:
        return 0
    else:
        return word_id - word_size + 1#word: 0; pad: 1; sep: 2; cls: 3; eof: 4
    
def word2sig_batch(word_size, word_id_batch, class_num=5):
    res = torch.zeros(*word_id_batch.size(), class_num).to(device)
    label = F.relu(word_id_batch - torch.tensor(word_size - 1).to(device))[:, :, None]
    res = res.scatter_(2, label, 1).to(device)
    return res

def construct_loss_mask(logit, first_sep, cls_location, max_len=512):
    loss_mask = []
    for i in range(len(first_sep)):
        loss_mask1 = torch.ones(max_len).to(device)
        loss_mask1[cls_location[i] + 1: ] = 0.0
        #loss_mask1[: cls_location[i] - 1] = 1.0
        loss_mask.append(loss_mask1)
    loss_mask = torch.stack(loss_mask, 0).to(device)
    total = loss_mask.sum()
    #print(loss_mask.size())
    return loss_mask[:, :, None].expand(-1, -1, logit.size()[2]), total

def construct_qkv_mask(cls_location, max_len=512):
    qkv_mask = []
    for cls_idx in cls_location:
        length = cls_idx + 1
        left = torch.ones(length)
        right = torch.zeros(max_len - length)
        qkv_mask1 = torch.cat([left, right], 0)
        qkv_mask.append(qkv_mask1)
    return torch.stack(qkv_mask, 0).to(device)

def construct_a_mask(first_sep, cls_location, max_len=512):#first not kl's sep
    a_mask = []
    for idx, cls_idx in zip(first_sep, cls_location):
        length = cls_idx + 1
        right_up = torch.ones(length, max_len - length)
        down = torch.ones(max_len - length, max_len)
        if idx + 1 != cls_idx:
            matrix1 = torch.zeros(length, idx + 1)
            matrix2 = torch.ones(idx + 1, length - (idx + 1))
            matrix3 = torch.triu(torch.ones(length - (idx + 1), length - (idx + 1)), diagonal=1)# not include I
            matrix23 = torch.cat([matrix2, matrix3], 0)
            left_up = torch.cat([matrix1, matrix23], 1)
        else:
            left_up = torch.zeros(length, length)
        up = torch.cat([left_up, right_up], 1)
        a_mask1 = torch.cat([up, down], 0)
        a_mask1[a_mask1 == 1.0] = 1e13
        #print(a_mask1)
        a_mask.append(a_mask1)
    return torch.stack(a_mask, 0).to(device)

def schedule_hpu(nlu_hpu, l_hs):
    decode_hpu = []
    st = 0
    for l_hs1 in l_hs:
        decode_hpu.append(nlu_hpu[st: st + l_hs1])
        st += l_hs1
    batch_location = [[] for _ in range(len(l_hs))]
    batch_data = []
    l_hs_matrix = [list(range(l_hs1)) for l_hs1 in l_hs]
    max_l_hs = max(l_hs)
    for i in range(len(l_hs_matrix)):
        if len(l_hs_matrix[i]) < max_l_hs:
            l_hs_matrix[i] += [-1 for _ in range(max_l_hs - len(l_hs_matrix[i]))]
    for i in range(max_l_hs):
        batch_data1 = []
        for j in range(len(l_hs)):
            if l_hs_matrix[j][i] != -1:
                batch_location[j].append((i, len(batch_data1)))
                batch_data1.append(decode_hpu[j][i])
        batch_data.append(batch_data1)
    return batch_data, batch_location

def Botnet_loss(logit, label, cls_idx, sep_decoder_idx, is_train=True, smooth=0.1):
    #print(logit[0].size())
    #print(label.size())
    loss_mask, total = construct_loss_mask(logit, sep_decoder_idx, cls_idx)
    #print(loss_mask.size())
    loss = F.log_softmax(logit, dim=2) * -1.0
    loss *= loss_mask
    #print(loss[0])
    nll_loss = loss.gather(dim=2, index=label.unsqueeze(2)).squeeze(2)
    if is_train:
        inf_mask = loss.eq(float('inf'))
        smooth_loss = loss.masked_fill(inf_mask, 0.).sum(dim=2)
        eps_i = smooth / (1.0 - inf_mask.float()).sum(dim=2)
        return nll_loss * (1. - smooth) + smooth_loss * eps_i
    else:
        return nll_loss

class Botnet(nn.Module):
    def __init__(self, w2id, id2w, word2sub, tf_emb_size, lS_lstm, dr, nb_tf_bo, hidden_tf_bo, heads, max_len=512, embed_dim=300):
        super(Botnet, self).__init__()
        self.emb_layer = Embedding_layer(w2id, id2w, word2sub, tf_emb_size, lS_lstm, dr, max_len, embed_dim)
        self.tf_body = Transformer_body(nb_tf_bo, tf_emb_size, hidden_tf_bo, dr, heads, tf_emb_size // heads, tf_emb_size // heads)
        self.classify_decoder = Classify_decoder(tf_emb_size, len(w2id))
        
    def forward(self, inputs, target, seg_idx, nlu_hpu, l_hs, cls_idx, sep_idx_encoder, sep_idx_decoder, mode=0, old_kl=None):
        x, label, kl_out = self.emb_layer(inputs, target, seg_idx, nlu_hpu, l_hs, mode=mode, old_kl=old_kl)
        mid_x = self.tf_body(x, cls_idx, sep_idx_decoder)
        y = self.classify_decoder(mid_x)
        
        return y, label, kl_out

class MultiHeaderAttention(nn.Module):
    def __init__(self, iS, heads, out_size, key_size=None):
        super(MultiHeaderAttention, self).__init__()
        self.heads = heads
        self.out_size = out_size
        self.out_dim = heads * out_size
        self.key_size = key_size if key_size else out_size
        self.iS = iS
        
        self.layerNorm = nn.LayerNorm(self.iS)
        self.q_dense = nn.Linear(self.iS, self.key_size * self.heads)
        self.k_dense = nn.Linear(self.iS, self.key_size * self.heads)
        self.v_dense = nn.Linear(self.iS, self.out_dim)
        self.o_dense = nn.Linear(self.out_dim, self.out_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, inputs, q_mask, kv_mask, a_mask):
        '''
        words, from 0 to w - 1, and then w is pad, w + 1 is sep, w + 2 is cls, w + 3 is eof
        
        previous cls will be the next of kl
        
        n tokens question
        m tokens answers
        k course title
        [kl, sep, 1, 2, 3, ..., n, sep, 1, 2, 3, ..., m, eof, sep, cls, pad, pad, ...], input no second eof, target will have it but no kl
        input: [kl, sep, 1, 2, 3, ..., n, sep, 1, 2, 3, ..., m, sep, cls, pad, pad, ...]
        target: [sep, 1, 2, 3, ..., n, sep, 1, 2, 3, ..., m, eof, sep, cls, pad, pad, ...]
        leading: [1, 2, 3, ..., k, sep, cls, pad, pad, ...]
        benefit is eof will only occur at target.
        q_mask: padding is zeros
        kv_mask: padding is zeros
        example:[I, am, a, student, !, [pad], [pad], [pad], [pad], [pad]] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        
        a_mask: language model mask with big penalty value
        '''
        q, k, v = inputs
        q = self.layerNorm(q)
        k = self.layerNorm(k)
        v = self.layerNorm(v)
        #q *= q_mask[:, :, None].expand_as(q)
        #k *= kv_mask[:, :, None].expand_as(k)
        #v *= kv_mask[:, :, None].expand_as(v)
        qw = self.q_dense(q)
        qw *= q_mask[:, :, None].expand_as(qw)#(bs, seq_len) -> (bs, seq_len, dim)
        kw = self.k_dense(k)
        kw *= kv_mask[:, :, None].expand_as(kw)
        vw = self.v_dense(v)
        vw *= kv_mask[:, :, None].expand_as(vw)
        
        qw = qw.view(*qw.size()[:-1], self.heads, self.key_size).permute(0, 2, 1, 3)#[bs, head, seq_len, dim]
        kw = kw.view(*kw.size()[:-1], self.heads, self.key_size).permute(0, 2, 1, 3)
        vw = vw.view(*vw.size()[:-1], self.heads, self.out_size).permute(0, 2, 1, 3)
        
        a = torch.einsum('bhqk,bhvk->bhqv', qw, kw) / torch.sqrt(torch.tensor(self.key_size).float().to(device))
        
        a -= a_mask[:, None, :, :].expand_as(a)
        
        a = self.softmax(a)
        
        o = torch.einsum('bhqv,bhvo->bhqo', a, vw)
        
        #o *= kv_mask[:, :, None].expand_as(o)
        o = self.o_dense(o.permute(0, 2, 1, 3).contiguous().view(*q.size()[:-1], self.out_dim))#[bs, h, seq_len, dim] -> [bs, seq_len, h, dim] -> [bs, seq_len, h*dim]
        o *= kv_mask[:, :, None].expand_as(o)
        #print(q.size())
        #print(v.size())
        #print(o.size())
        
        return o
    
class FFN(nn.Module):
    def __init__(self, o_dim, hS, dr):
        super(FFN, self).__init__()
        self.out_dim = o_dim
        self.hS = hS
        self.dr = dr
        
        self.dropout = nn.Dropout(self.dr)
        self.layerNorm = nn.LayerNorm(self.out_dim)
        self.fc1 = nn.Conv1d(self.out_dim, self.hS, 1)
        self.fc2 = nn.Conv1d(self.hS, self.out_dim, 1)
        
    def forward(self, q, o):
        o = self.dropout(o) + q
        q = o
        o = self.layerNorm(o)
        o = o.permute(0, 2, 1)
        o = self.fc1(o)
        o = gelu_new(o)
        o = self.fc2(o)
        o = o.permute(0, 2, 1)
        o = self.dropout(o) + q
        
        return o
    
class Transformer_block(nn.Module):
    def __init__(self, iS, hS, dr, heads, out_size, key_size=None):
        super(Transformer_block, self).__init__()
        self.atten = MultiHeaderAttention(iS, heads, out_size, key_size)
        self.ffn = FFN(heads * out_size, hS, dr)
        
    def forward(self, inputs, q_mask, kv_mask, a_mask):
        q, _, _ = inputs
        o = self.atten(inputs, q_mask, kv_mask, a_mask)
        o = self.ffn(q, o)
        
        return o# this is the next q for decoder, next qkv for encoder and next qkv for kl
    
class Transformer_encoder(nn.Module):#share parameter
    def __init__(self, lS, iS, hS, dr, heads, out_size, key_size=None):
        super(Transformer_encoder, self).__init__()
        self.lS = lS
        self.tf_block = Transformer_block(iS, hS, dr, heads, out_size, key_size)
        
    def forward(self, inputs, cls_idx, sep_idx):
        qkv_mask = construct_qkv_mask(cls_idx)
        a_mask = construct_a_mask(sep_idx, cls_idx)
        x = inputs
        for i in range(self.lS):
            x = self.tf_block([x, x, x], qkv_mask, qkv_mask, a_mask)
        
        return x

class Transformer_decoder(nn.Module):#share parameter
    def __init__(self, lS, iS, hS, dr, heads, out_size, key_size=None):
        super(Transformer_decoder, self).__init__()
        self.tf_block = Transformer_block(iS, hS, dr, heads, out_size, key_size)
        self.lS = lS
        
    def forward(self, inputs, target, cls_idx, sep_idx):
        qkv_mask = construct_qkv_mask(cls_idx)
        a_mask = construct_a_mask(sep_idx, cls_idx)
        #print(a_mask[0])
        x2 = inputs#from encoder
        x1 = target
        for i in range(self.lS):
            x1 = self.tf_block([x1, x2, x2], qkv_mask, qkv_mask, a_mask)
        
        return x1
    
class Transformer_body(nn.Module):
    def __init__(self, lS, iS, hS, dr, heads, out_size, key_size=None):
        super(Transformer_body, self).__init__()
        self.lS = lS
        self.tf_blocks = nn.ModuleList([Transformer_block(iS, hS, dr, heads, out_size, key_size) for _ in range(self.lS)])
        
    def forward(self, inputs, cls_idx, sep_idx):
        qkv_mask = construct_qkv_mask(cls_idx)
        a_mask = construct_a_mask(sep_idx, cls_idx)
        x = inputs
        for tf_block in self.tf_blocks:
            x = tf_block([x, x, x], qkv_mask, qkv_mask, a_mask)
        
        return x
    
class Classify_decoder(nn.Module):
    def __init__(self, tf_emb_size, word_size):
        super(Classify_decoder, self).__init__()
        self.word_size = word_size
        self.tf_emb_size = tf_emb_size
        self.out_layer = nn.Linear(self.tf_emb_size, self.word_size)
        
    def forward(self, target):
        logit = self.out_layer(target)
        
        return logit#[bs, seq_len, word_size]
    
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0, device=device) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq_len, bsz=None):
        pos_seq = torch.arange(0, pos_seq_len, 1.0, device=device).float()
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None,:,:].expand(bsz, -1, -1).to(device)
        else:
            return pos_emb[None,:,:].to(device)
        
class Embedding_layer(nn.Module):
    def __init__(self, w2id, id2w, word2sub, tf_emb_size, lS, dr, max_len=512, embed_dim=300):#word_emb is from 0 to word_size - 1
        super(Embedding_layer, self).__init__()
        self.word2sub = word2sub
        self.embed = nn.Embedding(len(w2id), embed_dim, padding_idx=len(w2id) - 4)
        self.w2id = w2id
        self.id2w = id2w
        self.tf_emb_size = tf_emb_size
        self.max_len = max_len
        self.lS = lS
        self.dr = dr
        self.emb_size = embed_dim
        
        self.pos_emb = PositionalEmbedding(tf_emb_size)
        
        self.emb_fc = nn.Linear(self.emb_size, self.tf_emb_size, False)
        
        self.enc_kl = nn.LSTM(input_size=self.emb_size, hidden_size=int(self.tf_emb_size / self.lS),
                             num_layers=self.lS, batch_first=True,
                             dropout=self.dr, bidirectional=True)
        self.cell_fc = nn.Linear(self.tf_emb_size, self.tf_emb_size * self.lS)
        self.hidden_fc = nn.Linear(self.tf_emb_size, self.tf_emb_size * self.lS)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        if self.embed.padding_idx is not None:
            with torch.no_grad():
                self.embed.weight[self.embed.padding_idx].fill_(0)
        
        
    def sentence2id(self, sentence):
        return [self.w2id[w] for w in sentence]
    '''
    def course_emb(self, course, seq_len, class_num=42):
        return torch.zeros(course.size()[0], class_num).scatter_(1, course[:, None], 1)[:, None, :].expand(-1, seq_len, -1).to(device)
        #return torch.zeros(course.size()[0], class_num).scatter_(1, course[:, None], 1).to(device)
    '''
    def forward(self, inputs, target, seg_idx, nlu_hpu, l_hs, mode=0, old_kl=None):#0 normal, 1 inference first, 2 inference next
        #print('inputs:', inputs[0])
        #print('target:', target[0])
        #print('nlu_hpu:', nlu_hpu[0])
        for i in range(len(nlu_hpu)):
            nlu_hpu[i] = self.sentence2id(nlu_hpu[i])
        for i in range(len(inputs)):
            inputs[i] = self.sentence2id(inputs[i])
        if mode == 0:
            for i in range(len(target)):
                target[i] = self.sentence2id(target[i])
        
        batch_data, batch_location = schedule_hpu(nlu_hpu, l_hs)
        #print('batch_data:', len(batch_data[0]))
        #print('batch_location:')
        #for e in batch_location:
        #    print(e)
        if mode == 0:
            batch_hs = []
            for b in range(len(batch_data)):
                batch_hs1 = []
                for ib in range(len(batch_data[b])):
                    batch_hs1.append(len(batch_data[b][ib]))
                    batch_data[b][ib] = torch.tensor(batch_data[b][ib]).to(device)
                batch_hs.append(batch_hs1)
                max_cur = max(batch_hs1)
                for ib in range(len(batch_data[b])):
                    #print(batch_data[b][ib])
                    cur_len = batch_data[b][ib].size()[0]
                    if cur_len < max_cur:
                        pad = torch.zeros(max_cur - cur_len).long().to(device)
                        batch_data[b][ib] = torch.cat([batch_data[b][ib], pad], 0)
                batch_data[b] = self.embed(torch.stack(batch_data[b], 0).to(device))
            kl = encode(self.enc_kl, batch_data[0], batch_hs[0], return_hidden=False, hc0=None, last_only=True).squeeze(1)
            #print(0, kl.size())
            #print(kl.size())
            if len(batch_data) > 1:
                for b in range(1, len(batch_data)):
                    tmp_h = []
                    tmp_c = []
                    for j, batch_location1 in enumerate(batch_location):
                        if b < len(batch_location1):
                            #print(j)
                            tmp_h.append(kl[j])
                            tmp_c.append(kl[j])
                    tmp_h = torch.stack(tmp_h, 0).to(device)
                    tmp_c = torch.stack(tmp_c, 0).to(device)
                    hidden = self.hidden_fc(tmp_h)
                    hidden = hidden.view(hidden.size()[0], self.lS * 2, self.tf_emb_size // 2).transpose(0, 1).contiguous()
                    cell = self.cell_fc(tmp_c)
                    cell = cell.view(cell.size()[0], self.lS * 2, self.tf_emb_size // 2).transpose(0, 1).contiguous()
                    tmp_kl = encode(self.enc_kl, batch_data[b], batch_hs[b], return_hidden=False, hc0=(hidden, cell), last_only=True).squeeze(1)
                    #print(b, tmp_kl.size())
                    for j, batch_location1 in enumerate(batch_location):
                        if b < len(batch_location1):
                            kl[j] = tmp_kl[batch_location1[b][1]]
        elif mode == 1:
            batch_hs = []
            for b in range(len(batch_data)):
                batch_hs1 = []
                for ib in range(len(batch_data[b])):
                    batch_hs1.append(len(batch_data[b][ib]))
                    batch_data[b][ib] = torch.tensor(batch_data[b][ib]).to(device)
                batch_hs.append(batch_hs1)
                max_cur = max(batch_hs1)
                for ib in range(len(batch_data[b])):
                    #print(batch_data[b][ib])
                    cur_len = batch_data[b][ib].size()[0]
                    if cur_len < max_cur:
                        pad = torch.zeros(max_cur - cur_len).long().to(device)
                        batch_data[b][ib] = torch.cat([batch_data[b][ib], pad], 0)
                batch_data[b] = self.embed(torch.stack(batch_data[b], 0).to(device))
            kl = encode(self.enc_kl, batch_data[0], batch_hs[0], return_hidden=False, hc0=None, last_only=True).squeeze(1)
        else:
            batch_hs = []
            for b in range(len(batch_data)):
                batch_hs1 = []
                for ib in range(len(batch_data[b])):
                    batch_hs1.append(len(batch_data[b][ib]))
                    batch_data[b][ib] = torch.tensor(batch_data[b][ib]).to(device)
                batch_hs.append(batch_hs1)
                max_cur = max(batch_hs1)
                for ib in range(len(batch_data[b])):
                    #print(batch_data[b][ib])
                    cur_len = batch_data[b][ib].size()[0]
                    if cur_len < max_cur:
                        pad = torch.zeros(max_cur - cur_len).long().to(device)
                        batch_data[b][ib] = torch.cat([batch_data[b][ib], pad], 0)
                batch_data[b] = self.embed(torch.stack(batch_data[b], 0).to(device))
            hidden = self.hidden_fc(old_kl)
            hidden = hidden.view(hidden.size()[0], self.lS * 2, self.tf_emb_size // 2).transpose(0, 1).contiguous()
            cell = self.cell_fc(old_kl)
            cell = cell.view(cell.size()[0], self.lS * 2, self.tf_emb_size // 2).transpose(0, 1).contiguous()
            kl = encode(self.enc_kl, batch_data[b], batch_hs[b], return_hidden=False, hc0=(hidden, cell), last_only=True).squeeze(1)
        
        #print('kl:', kl.size())
        
        inputs = torch.tensor(inputs).to(device)
        #print(inputs.size())
        target_label = torch.tensor(target).to(device) if mode == 0 else target
        #print(seg_idx[0])
        seg_emb = torch.tensor(seg_idx).float().to(device)
        
        #inputs_sig = self.sig_fc(word2sig_batch(self.word_size, inputs).to(device))
        #print(inputs_sig.size())
        #target_sig = word2sig_batch(self.word_size, target_label).to(device)
        
        inputs = self.emb_fc(self.embed(inputs))
        #print(inputs[0])
        #target = self.word_emb[target].to(device)
        
        #inputs += inputs_sig
        #target += target_sig
        
        #print(c_emb.size())
        #kl = self.course_fc(c_emb)
        #print(c_emb.size())
        
        kl_out = kl
        
        kl = kl[:, None, :]
        
        inputs = torch.cat([kl, inputs], 1)
        
        #print(inputs.size())
        
        p_emb_i = self.pos_emb(inputs.size()[1], inputs.size()[0])
        #p_emb_t = self.pos_emb(target.size()[1], target.size()[0])
        
        inputs += p_emb_i
        #print(inputs.size())
        inputs += seg_emb[:, :, None].expand_as(inputs)
        #target += p_emb_t
        
        return inputs, target_label, kl_out