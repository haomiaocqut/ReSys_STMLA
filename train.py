import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import numpy as np
import torch
from collections import defaultdict
import gc
import os
from math import radians, cos, sin, asin, sqrt
from collections import deque,Counter
from torch.nn import Parameter
import math
import torch.optim as optim
from enum import IntEnum
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

print(torch.cuda.is_available())


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(0.5)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        print('type(attention_scores)',type(attention_scores))
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MogLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, mog_iterations: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.mog_iterations = mog_iterations
        # Define/initialize all tensors
        self.Wih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))
        # Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self, xt, ht):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht

    # Define forward pass through all LSTM cells across all timesteps.
    # By using PyTorch functions, we get backpropagation for free.
    # def forward(self, x: torch.Tensor,
    #             init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    #             ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    def forward(self, x,init_states):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        # ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
            # print('Ct.shape', Ct.shape)   #torch.Size([1, 64, 500])
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            xt, ht = self.mogrify(xt, ht)  # mogrification
            # print('xt.shape', xt.shape)   #torch.Size([1, 64, 500])
            # print('ht.shape', ht.shape)   #torch.Size([1, 64, 500])
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            # print('gates.shape', gates.shape)   #gates.shape torch.Size([1, 64, 2000])
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)

            ### The LSTM Cell!
            ft = torch.sigmoid(forgetgate)
            # print('ft.shape',ft.shape)
            it = torch.sigmoid(ingate)
            # print('it.shape', it.shape)
            Ct_candidate = torch.tanh(cellgate)
            # print('Ct_candidate.shape', Ct_candidate.shape)
            ot = torch.sigmoid(outgate)
            # print('ot.shape', ot.shape)
            # outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            # print('Ct',Ct)
            ht = ot * torch.tanh(Ct)
            ###

            hidden_seq.append(ht.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (ht, Ct)
        #@表示矩阵相乘，用*表示按元素相乘


class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i + 1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 128)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def pad_batch_of_lists_masks(batch_of_lists, max_len):
    padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists]
    padded_mask = [[1.0]*(len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padde_mask_non_local = [[1.0] * (len(l)) + [0.0] * (max_len - len(l)) for l in batch_of_lists]
    return padded, padded_mask, padde_mask_non_local

def pad_batch_of_lists_masks_test(batch_of_lists, max_len):
    padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists]
    padded2 = [l[:-1] + [0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padded_mask = [[0.0]*(len(l) - 2) + [1.0] + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padde_mask_non_local = [[1.0] * (len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    return padded, padded2, padded_mask, padde_mask_non_local

class Model(nn.Module):
    def __init__(self, n_users, n_items, emb_size=500, hidden_units=500, dropout=0.8, user_dropout=0.5, data_neural = None, tim_sim_matrix = None,num_attention_heads=2):
        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_units = hidden_units
        if emb_size == None:
            emb_size = hidden_units
        self.emb_size = emb_size
        ## todo why embeding?
        self.item_emb = nn.Embedding(n_items, emb_size)
        self.emb_tim = nn.Embedding(48, 10)
        self.lstmcell = MogLSTM(input_sz=emb_size, hidden_sz=hidden_units, mog_iterations=4)
        self.lstmcell_history=MogLSTM(input_sz=emb_size, hidden_sz=hidden_units, mog_iterations=4)
        self.linear = nn.Linear(hidden_units*2 , n_items)
        self.dropout = nn.Dropout(0.3)
        self.user_dropout = nn.Dropout(user_dropout)
        self.data_neural = data_neural
        self.tim_sim_matrix = tim_sim_matrix
        self.dilated_rnn = MogrifierLSTMCell(input_size=emb_size, hidden_size=hidden_units, mogrify_steps=4)
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        # self.SelfAttention = SelfAttention(self.num_attention_heads, self.input_size, self.hidden_units,self.hidden_dropout_prob)
        # self.SelfAttention.cuda()
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_units,1)  # hidden : [batch_size, hidden_units * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.tanh(torch.bmm(lstm_output, hidden).squeeze(2))  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, hidden_units * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size,hidden_units * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # return context,torch.from_numpy(soft_attn_weights.data.numpy())  # context : [batch_size, n_hidden * num_directions(=2)]  attention : [batch_size, n_step]
        return context, soft_attn_weights  # context : [batch_size, n_hidden * num_directions(=2)]  attention : [batch_size, n_step]

    def forward(self, user_vectors, item_vectors, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, is_train, poi_distance_matrix, sequence_dilated_rnn_index_batch):
        batch_size = item_vectors.size()[0]
        sequence_size = item_vectors.size()[1]
        items = self.item_emb(item_vectors)
        item_vectors = item_vectors.cpu()
        x = items
        h1 = Variable(torch.zeros(1,batch_size, self.hidden_units)).cuda()
        c1 = Variable(torch.zeros(1,batch_size, self.hidden_units)).cuda()
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        out = out.squeeze(0)

        print('out',out.shape)  #out torch.Size([12, 64, 500])
        out = out.transpose(0, 1)#batch_size * sequence_length * embedding_dim

        # ##############################
        attn_output, attention = self.attention_net(out, h1)
        attn_output = attn_output.unsqueeze(1)
        attention = attention.unsqueeze(1).transpose(1, 2)
        out = attention.matmul(attn_output)
        out = F.selu(out)
        # out = self.dropout(out)
        ########################################
        x1 = items
        # ###########################################################
        user_batch = np.array(user_vectors.cpu())
        y_list = []
        out_hie = []
        for ii in range(batch_size):
            ##########################################
            current_session_input_dilated_rnn_index = sequence_dilated_rnn_index_batch[ii]
            hiddens_current = x1[ii]
            dilated_lstm_outs_h = []
            dilated_lstm_outs_c = []
            for index_dilated in range(len(current_session_input_dilated_rnn_index)):
                index_dilated_explicit = current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0)
                if index_dilated == 0:
                    h = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    c = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    (h, c) = self.dilated_rnn(hidden_current, (h, c))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dilated_rnn(hidden_current, (dilated_lstm_outs_h[index_dilated_explicit], dilated_lstm_outs_c[index_dilated_explicit]))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
            dilated_lstm_outs_h.append(hiddens_current[len(current_session_input_dilated_rnn_index):])
            dilated_out = torch.cat(dilated_lstm_outs_h, dim = 0).unsqueeze(0)
            out_hie.append(dilated_out)
            user_id_current = user_batch[ii]
            current_session_timid = sequence_tim_batch[ii][:-1]
            current_session_poiid = item_vectors[ii][:len(current_session_timid)]
            session_id_current = session_id_batch[ii]
            current_session_embed = out[ii]
            current_session_mask = mask_batch_ix_non_local[ii].unsqueeze(1)
            sequence_length = int(sum(np.array(current_session_mask.cpu()))[0])
            current_session_represent_list = []
            if is_train:
                for iii in range(sequence_length-1):
                    current_session_represent = torch.sum(current_session_embed * current_session_mask, dim=0).unsqueeze(0)/sum(current_session_mask)
                    current_session_represent_list.append(current_session_represent)
            else:
                for iii in range(sequence_length-1):
                    current_session_represent_rep_item = current_session_embed[0:iii+1]
                    current_session_represent_rep_item = torch.sum(current_session_represent_rep_item, dim = 0).unsqueeze(0)/(iii + 1)
                    current_session_represent_list.append(current_session_represent_rep_item)

            current_session_represent = torch.cat(current_session_represent_list, dim = 0)
            list_for_sessions = []
            list_for_avg_distance = []
            h2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()###whole sequence
            c2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()
            for jj in range(session_id_current):
                sequence = [s[0] for s in self.data_neural[user_id_current]['sessions'][jj]]
                sequence = Variable(torch.LongTensor(np.array(sequence))).cuda()
                sequence_emb = self.item_emb(sequence).unsqueeze(1)   ## sequence_emb:torch.Size([8, 1, 500])
                sequence_emb = sequence_emb.transpose(0,1)
                sequence = sequence.cpu()
                sequence_emb, (h2, c2) = self.lstmcell_history(sequence_emb, (h2, c2))
                sequence_emb = sequence_emb.squeeze(0)
                # ###################################  #
                out1 = SelfAttention(self.num_attention_heads, self.input_size, self.hidden_units).cuda()
                sequence_emb = out1(sequence_emb)
                # ###################################
                sequence_tim_id = [s[1] for s in self.data_neural[user_id_current]['sessions'][jj]]
                jaccard_sim_row = Variable(torch.FloatTensor(self.tim_sim_matrix[current_session_timid]),requires_grad=False).cuda()
                # jaccard_sim_row = Variable(torch.FloatTensor(current_session_timid),requires_grad=False).cuda()
                jaccard_sim_expicit = jaccard_sim_row[:,sequence_tim_id]
                distance_row = poi_distance_matrix[current_session_poiid]
                distance_row_expicit = Variable(torch.FloatTensor(distance_row[:,sequence]),requires_grad=False).cuda()
                distance_row_expicit_avg = torch.mean(distance_row_expicit, dim = 1)
                jaccard_sim_expicit_last = F.softmax(jaccard_sim_expicit)
                hidden_sequence_for_current1 = torch.mm(jaccard_sim_expicit_last, sequence_emb.squeeze(1))
                hidden_sequence_for_current =  hidden_sequence_for_current1
                list_for_sessions.append(hidden_sequence_for_current.unsqueeze(0))
                list_for_avg_distance.append(distance_row_expicit_avg.unsqueeze(0))
            avg_distance = torch.cat(list_for_avg_distance, dim = 0).transpose(0,1)
            sessions_represent = torch.cat(list_for_sessions, dim=0).transpose(0,1) ##current_items * history_session_length * embedding_size
            current_session_represent = current_session_represent.unsqueeze(2) ### current_items * embedding_size * 1
            sims = F.softmax(sessions_represent.bmm(current_session_represent).squeeze(2)* 1.0/avg_distance, dim = 1).unsqueeze(1) ##==> current_items * 1 * history_session_length
            out_y_current = sims.bmm(sessions_represent).squeeze(1)
            out_y_current =torch.selu(self.linear1(sims.bmm(sessions_represent).squeeze(1)))
            out_y_current_padd = Variable(torch.FloatTensor(sequence_size - sequence_length + 1, self.emb_size).zero_(),
                                          requires_grad=False).cuda()
            out_layer_2_list = []
            out_layer_2_list.append(out_y_current)
            out_layer_2_list.append(out_y_current_padd)
            out_layer_2 = torch.cat(out_layer_2_list, dim=0).unsqueeze(0)
            y_list.append(out_layer_2)
        y = torch.selu(torch.cat(y_list,dim=0))
        out_hie = F.selu(torch.cat(out_hie, dim = 0))
        out = F.selu(out)
        out = (out + out_hie) * 0.5
        out_put_emb_v1 = torch.cat([y, out], dim=2)
        output_ln = self.linear(out_put_emb_v1)
        output = F.log_softmax(output_ln, dim=-1)
        return output




def caculate_time_sim(data_neural):
    time_checkin_set = defaultdict(set)
    for uid in data_neural:
        uid_sessions = data_neural[uid]
        for sid in uid_sessions['sessions']:
            session_current = uid_sessions['sessions'][sid]
            for checkin in session_current:
                timid = checkin[1]
                locid = checkin[0]
                if timid not in time_checkin_set:
                    time_checkin_set[timid] = set()
                time_checkin_set[timid].add(locid)
    sim_matrix = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            set_i = time_checkin_set[i]
            set_j = time_checkin_set[j]
            # jaccard_ij = len(set_i & set_j)/len(set_i | set_j)
            jaccard_ij = len(set_i & set_j)+1
            jaccard_ij /= math.sqrt((len(set_i)+1) * (len(set_j)+1))
            sim_matrix[i][j] = jaccard_ij
    return sim_matrix

def caculate_poi_distance(poi_coors):
    print("distance matrix")
    sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
    for i in range(len(poi_coors)):
        for j in range(i , len(poi_coors)):
            poi_current = i + 1
            poi_target = j + 1
            poi_current_coor = poi_coors[poi_current]
            poi_target_coor = poi_coors[poi_target]
            distance_between = geodistance(poi_current_coor[1], poi_current_coor[0], poi_target_coor[1], poi_target_coor[0])
            if distance_between<1:
                distance_between = 1
            sim_matrix[poi_current][poi_target] = distance_between
            sim_matrix[poi_target][poi_current] = distance_between
    pickle.dump(sim_matrix, open('0user_distance.pkl', 'wb'))
    return sim_matrix

def generate_input_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]]), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([s[1] for s in session[:-1]]), (len(session[:-1]), 1))
            target = np.array([s[0] for s in session[1:]])
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['target'] = Variable(torch.LongTensor(target))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: x[1], reverse=False)
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count
            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target))
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())
    train_queue = list()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def create_dilated_rnn_input(session_sequence_current, poi_distance_matrix):
    sequence_length = len(session_sequence_current)
    session_sequence_current.reverse()
    session_dilated_rnn_input_index = [0] * sequence_length
    for i in range(sequence_length - 1):
        current_poi = [session_sequence_current[i]]
        poi_before = session_sequence_current[i + 1 :]
        distance_row = poi_distance_matrix[current_poi]
        distance_row_explicit = distance_row[:, poi_before][0]
        index_closet = np.argmin(distance_row_explicit)
        session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length-2-index_closet-i
    session_sequence_current.reverse()
    return session_dilated_rnn_input_index



def generate_detailed_batch_data(one_train_batch):
    session_id_batch = []
    user_id_batch = []
    sequence_batch = []
    sequences_lens_batch = []
    sequences_tim_batch = []
    sequences_dilated_input_batch = []
    for sample in one_train_batch:
        user_id_batch.append(sample[0])
        session_id_batch.append(sample[1])
        session_sequence_current = [s[0] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_tim_current = [s[1] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_dilated_input = create_dilated_rnn_input(session_sequence_current, poi_distance_matrix)
        sequence_batch.append(session_sequence_current)
        sequences_lens_batch.append(len(session_sequence_current))
        sequences_tim_batch.append(session_sequence_tim_current)
        sequences_dilated_input_batch.append(session_sequence_dilated_input)
    return user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequences_tim_batch, sequences_dilated_input_batch


def train_network(network, num_epoch=55 ,batch_size =64,criterion = None):
    candidate = data_neural.keys()
    data_train, train_idx = generate_input_history(data_neural, 'train', candidate=candidate)
    for epoch in range(num_epoch):
        network.train(True)
        i = 0
        run_queue = generate_queue(train_idx, 'random', 'train')
        for one_train_batch in minibatch(run_queue, batch_size = batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(one_train_batch)
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network(user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, True, poi_distance_matrix, sequence_dilated_rnn_index_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()
            # train with backprop
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            opt.step()
            if (i + 1) % 20 == 0:
                print("epoch" + str(epoch) + ": loss: " + str(loss))
            i += 1
        results = evaluate(network, 1)
        print("Scores: ", results)


def get_acc(target, scores):
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return acc.tolist(), ndcg.tolist()

def evaluate(network, batch_size = 2):
    network.train(False)
    candidate = data_neural.keys()
    data_test, test_idx = generate_input_long_history(data_neural, 'test', candidate=candidate)
    users_acc = {}
    with torch.no_grad():
        run_queue = generate_queue(test_idx, 'normal', 'test')
        for one_test_batch in minibatch(run_queue, batch_size=batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(
                one_test_batch)
            user_id_batch_test = user_id_batch
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network(user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, False, poi_distance_matrix, sequence_dilated_rnn_index_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            for ii, u_current in enumerate(user_id_batch_test):
                if u_current not in users_acc:
                    users_acc[u_current] = [0, 0, 0, 0, 0, 0, 0]
                acc, ndcg = get_acc(actual_next_tokens[ii], predictions_logp[ii])
                users_acc[u_current][1] += acc[2][0]#@1
                users_acc[u_current][2] += acc[1][0]#@5
                users_acc[u_current][3] += acc[0][0]#@10
                ###ndcg
                users_acc[u_current][4] += ndcg[2][0]  # @1
                users_acc[u_current][5] += ndcg[1][0]  # @5
                users_acc[u_current][6] += ndcg[0][0]  # @10
                users_acc[u_current][0] += (sequences_lens_batch[ii]-1)
        tmp_acc = [0.0,0.0,0.0, 0.0, 0.0, 0.0]##last 3 ndcg
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
            tmp_acc[2] = users_acc[u][3] + tmp_acc[2]

            tmp_acc[3] = users_acc[u][4] + tmp_acc[3]
            tmp_acc[4] = users_acc[u][5] + tmp_acc[4]
            tmp_acc[5] = users_acc[u][6] + tmp_acc[5]
            sum_test_samples = sum_test_samples + users_acc[u][0]
        avg_acc = (np.array(tmp_acc)/sum_test_samples).tolist()
        return avg_acc

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance   #半正矢公式  Haversine公式


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data = pickle.load(open('.pkl', 'rb'), encoding='iso-8859-1')
    vid_list = data['vid_list']
    uid_list = data['uid_list']
    data_neural = data['data_neural']
    poi_coordinate = data['vid_lookup']
    loc_size = len(vid_list)
    uid_size = len(uid_list)
    time_sim_matrix = caculate_time_sim(data_neural)
    poi_distance_matrix = caculate_poi_distance(poi_coordinate)
    # poi_distance_matrix = pickle.load(open('.pkl', 'rb'), encoding='iso-8859-1')
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    n_users = uid_size
    n_items = loc_size
    session_id_sequences = None
    user_id_session = None
    network = Model(n_users=n_users, n_items=n_items, data_neural=data_neural, tim_sim_matrix=time_sim_matrix).to(
        device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=0.00005,
                               weight_decay=1 * 1e-6)
    criterion = nn.NLLLoss().cuda()
    train_network(network,criterion=criterion)
