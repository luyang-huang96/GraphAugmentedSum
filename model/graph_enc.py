import torch
import math
from torch import nn
from torch.nn import functional as F
from model.util import len_mask
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

# code from: https://github.com/rikdz/GraphWriter
class MultiHeadAttention_old(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim,requires_grad=False).float()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)
        self.ln = nn.LayerNorm(num_units)

    def forward(self, query, keys, mask=None):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim).cuda()

        if mask is not None:
            mask = mask.repeat(self._h,1,1)
            attention.masked_fill_(mask,-float('inf'))
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        #attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)
        # apply layer normalization
        #attention = self.ln(attention)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=4,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(key_dim,requires_grad=False).float()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)


    def forward(self, query, keys, mask=None, debug=False):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.stack(Q.split(split_size=chunk_size, dim=-1))
        K = torch.stack(K.split(split_size=chunk_size, dim=-1))
        V = torch.stack(V.split(split_size=chunk_size, dim=-1))

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim).cuda()

        if mask is not None:
            mask = mask.unsqueeze(0).repeat(self._h,1,1)
            attention.masked_fill_(mask,-float('inf'))
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2).squeeze(0)
        # residual connection
        # attention += query
        # apply batch normalization
        #attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)
        # apply layer normalization
        #attention = self.ln(attention)

        attention = F.relu(attention)

        return attention


class Block_old(nn.Module):
    def __init__(self,args):
        super().__init__()
        hidden_size = args.get('graph_hsz', 128)
        blockdrop = args.get('blockdrop', 0.1)


        self.attn = MultiHeadAttention(hidden_size, hidden_size, hidden_size, h=4, dropout_p=blockdrop)
        self.l1 = nn.Linear(hidden_size, hidden_size*4)
        self.l2 = nn.Linear(hidden_size*4, hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(blockdrop)
        #self.act = gelu
        self.act = nn.PReLU(hidden_size*4)
        self.gatact = nn.PReLU(hidden_size)

    def forward(self,q,k,m):
        q = self.attn(q,k,mask=m).squeeze(1)
        t = self.ln_1(q)
        q = self.drop(self.l2(self.act(self.l1(t))))
        q = self.ln_2(q+t)
        return q

class Block(nn.Module):
    def __init__(self,args):
        super().__init__()
        hidden_size = args.get('graph_hsz', 128)
        blockdrop = args.get('blockdrop', 0.1)


        self.attn = MultiHeadAttention(hidden_size, hidden_size, hidden_size, h=4, dropout_p=blockdrop)
        # self.l1 = nn.Linear(hidden_size, hidden_size*4)
        # self.l2 = nn.Linear(hidden_size*4, hidden_size)
        # self.ln_1 = nn.LayerNorm(hidden_size)
        # self.ln_2 = nn.LayerNorm(hidden_size)
        # self.drop = nn.Dropout(blockdrop)
        # #self.act = gelu
        # self.act = nn.PReLU(hidden_size*4)
        # self.gatact = nn.PReLU(hidden_size)

    def forward(self,q,k,m):
        # q: n * d k: r * d m: n * r
        q = self.attn(q,k,mask=m)
        # t = self.ln_1(q)
        # q = self.drop(self.l2(self.act(self.l1(t))))
        # q = self.ln_2(q+t)
        return q

class graph_encode(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        rtoks = args.get('rtoks', 1)
        hidden_size = args.get('graph_hsz', 300)
        blockdrop = args.get('blockdrop', 0.1)
        prop = args.get('prop', 6)
        sparse = args.get('sparse', False)
        model = args.get('graph_model', 'transformer')

        self.renc = nn.Embedding(rtoks, hidden_size)
        nn.init.xavier_normal_(self.renc.weight)

        if model == "gat":
            self.gat = nn.ModuleList([MultiHeadAttention(hidden_size, hidden_size, hidden_size, h=4, dropout_p=blockdrop) for _ in range(prop)])
        else:
            self.gat = nn.ModuleList([Block(args) for _ in range(prop)])

        self._pad_entity = nn.Parameter(torch.Tensor(1, hidden_size))
        nn.init.uniform_(self._pad_entity)


        self.prop = prop
        self.sparse = sparse
        self.model = model

    def pad(self,tensor,length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self,adjs,rels,ents):
        # print('adjs', len(adjs))
        # for adj in adjs:
        #   print('adj:', adj.size())
        #   print(adj.sum())
        # print('rels', rels)
        # for ent in ents:
        #   print('ents', ent.size())
        # print(ents[1])
        vents, entlens = ents
        # if self.args.entdetach:
        #   vents = torch.tensor(vents,requires_grad=False)
        vrels = [self.renc(x) for x in rels]
        glob = []
        graphs = []
        for i,adj in enumerate(adjs):
            vgraph = torch.cat((self._pad_entity, vents[i][:entlens[i]], vrels[i]),0)
            N = vgraph.size(0)
            if self.sparse:
                lens = [len(x) for x in adj]
                m = max(lens)
                mask = torch.arange(0,m).unsqueeze(0).repeat(len(lens),1).long()
                mask = (mask <= torch.LongTensor(lens).unsqueeze(1)).cuda()
                mask = (mask == 0).unsqueeze(1)
            else:
                mask = (adj == 0).unsqueeze(1)
            for j in range(self.prop):
                if self.sparse:
                    ngraph = [vgraph[k] for k in adj]
                    ngraph = [self.pad(x,m) for x in ngraph]
                    ngraph = torch.stack(ngraph,0)
                    #print(ngraph.size(),vgraph.size(),mask.size())
                    vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
                else:
                    ngraph = torch.tensor(vgraph.repeat(N,1).view(N,N,-1),requires_grad=False)
                    vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
                    if math.isnan(vgraph.sum()):
                        print('vgraph:', vgraph)
                        print('mask:', mask)
                        print('adj', adj)
                        print(j, i)
                    if self.model == 'gat':
                        vgraph = vgraph.squeeze(1)
                        vgraph = self.gatact(vgraph)
            graphs.append(vgraph)
            glob.append(vgraph[entlens[i]])
        elens = [x.size(0) for x in graphs]
        gents = [self.pad(x,max(elens)) for x in graphs]
        gents = torch.stack(gents,0)
        elens = torch.LongTensor(elens)
        emask = torch.arange(0,gents.size(1)).unsqueeze(0).repeat(gents.size(0),1).long()
        emask = (emask <= elens.unsqueeze(1)).cuda()
        glob = torch.stack(glob,0)
        if math.isnan(gents.sum()):
            for gent in gents:
                if math.isnan(gent.sum()):
                    print('gent:', gent)
        return None, glob ,(gents, emask)

# class gat_encode(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         rtoks = args.get('rtoks', 1)
#         hidden_size = args.get('graph_hsz', 300)
#         blockdrop = args.get('blockdrop', 0.1)
#         prop = args.get('prop', 1)
#         sparse = args.get('sparse', False)
#         model = args.get('graph_model', 'transformer')
#         adj_type = args.get('adj_type', 'concat_triple')
#         self._adj_type = adj_type
#
#         if self._adj_type == 'concat_triple':
#             self._rproj = nn.Linear(hidden_size*3, hidden_size)
#             self._rrelu = nn.LeakyReLU()
#         self._rtrans = nn.Linear(hidden_size, hidden_size)
#
#         self.gat = Block(args)
#         # if model == "gat":
#         #     self.gat = nn.ModuleList([MultiHeadAttention(hidden_size, hidden_size, hidden_size, h=4, dropout_p=blockdrop) for _ in range(prop)])
#         # else:
#         #     self.gat = nn.ModuleList([Block(args) for _ in range(prop)])
#
#         self._pad_entity = nn.Parameter(torch.Tensor(1, hidden_size))
#         nn.init.uniform_(self._pad_entity)
#
#
#         self.prop = prop
#         self.sparse = sparse
#         self.model = model
#
#     def pad(self,tensor,length):
#         return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])
#
#     def forward(self, adjs, triples, nodes, node_num, relations):
#         # input triples B*(R*3d)  R*d
#         triple_outs = []
#         for _i, adj in enumerate(adjs):
#             if self._adj_type == 'concat_triple':
#                 triple = self._rrelu(self._rproj(triples[_i])) # R * d
#                 R = triple.size(0)
#                 N = node_num[_i]
#                 ngraph = nodes[_i, :N, :] # N * d
#                 mask = (adj == 0) # N * R
#                 # triple_out = self.gat(ngraph, triple, mask)
#                 triple_out = triple
#             else:
#                 N = node_num[_i]
#                 ngraph = nodes[_i, :N, :]  # N * d
#                 mask = (adj == 0)  # N * N
#                 triple_out = self.gat(ngraph, ngraph, mask)
#             triple_outs.append(triple_out)
#         max_n= max(node_num)
#         nodes = torch.stack(
#                 [torch.cat([s, torch.zeros(max_n - n, s.size(1)).to(s.device)], dim=0)
#                  if n != max_n
#                  else s
#                  for s, n in zip(triple_outs, node_num)],
#                 dim=0
#             )
#         # relations = self._rtrans(relations)
#
#         return nodes, relations

class gat_encode(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        rtoks = args.get('rtoks', 1)
        hidden_size = args.get('graph_hsz', 300)
        blockdrop = args.get('blockdrop', 0.1)
        prop = args.get('prop', 1)
        sparse = args.get('sparse', False)
        model = args.get('graph_model', 'gat')
        adj_type = args.get('adj_type', 'edge_as_node')
        self._adj_type = adj_type
        self._model = model

        if self._adj_type == 'concat_triple':
            self._rproj = nn.Linear(hidden_size*3, hidden_size)
            self._rrelu = nn.LeakyReLU()
        self._rtrans = nn.Linear(hidden_size, hidden_size)

        print('graph model:', model)
        if model == 'ggnn':
            self.ggnn = GGNN(hidden_size, 1, prop)
        elif model == 'gcn':
            self.gcn = GCN(hidden_size, blockdrop)
        else:
            self.gat = Block(args)
        # if model == "gat":
        #     self.gat = nn.ModuleList([MultiHeadAttention(hidden_size, hidden_size, hidden_size, h=4, dropout_p=blockdrop) for _ in range(prop)])
        # else:
        #     self.gat = nn.ModuleList([Block(args) for _ in range(prop)])

        # self._pad_entity = nn.Parameter(torch.Tensor(1, hidden_size))
        # nn.init.uniform_(self._pad_entity)


        self.prop = prop
        self.sparse = sparse
        self.model = model

    def pad(self,tensor,length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])

    def forward(self, adjs, triples, nodes, node_num, relations):
        # input triples B*(R*3d)  R*d
        if self._model == 'ggnn':
            return self.ggnn(triples, adjs)
        elif self._model == 'gcn':
            return self.gcn(triples, adjs)
        triple_outs = []
        for _i, adj in enumerate(adjs):
            if self._adj_type == 'concat_triple':
                triple = self._rrelu(self._rproj(triples[_i])) # R * d
                R = triple.size(0)
                N = node_num[_i]
                ngraph = nodes[_i, :N, :] # N * d
                mask = (adj == 0) # N * R
                # triple_out = self.gat(ngraph, triple, mask)
                triple_out = triple
            else:
                N = node_num[_i]
                ngraph = nodes[_i, :N, :]  # N * d
                mask = (adj == 0)  # N * N
                triple_out = self.gat(ngraph, ngraph, mask)
            triple_outs.append(triple_out)
        max_n= max(node_num)
        nodes = torch.stack(
                [torch.cat([s, torch.zeros(max_n - n, s.size(1)).to(s.device)], dim=0)
                 if n != max_n
                 else s
                 for s, n in zip(triple_outs, node_num)],
                dim=0
            )
        # relations = self._rtrans(relations)

        return nodes, relations


class node_mask(nn.Module):
    def __init__(self, mask_type, emb_dim=128, feat_dim=128):
        super().__init__()
        self._type = mask_type
        if mask_type == 'gold' or mask_type == 'none':
            self._mask = None
        elif mask_type == 'soft':
            self._mask = nn.Linear(emb_dim, 1)
        elif mask_type == 'soft+sent':
            self._wc = nn.Parameter(torch.Tensor(feat_dim, feat_dim))
            self._wh = nn.Parameter(torch.Tensor(emb_dim, feat_dim))
            self._v = nn.Parameter(torch.Tensor(feat_dim))
            torch.nn.init.xavier_normal_(self._wc)
            torch.nn.init.xavier_normal_(self._wh)
            torch.nn.init.uniform_(self._v, -1e-2, 1e-2)
            #self._bi_attn = nn.Bilinear(emb_dim, feat_dim, 1)
            self._bi_attn = nn.Parameter(torch.Tensor(emb_dim, feat_dim))
            torch.nn.init.xavier_normal_(self._bi_attn)
            # final selection




    def forward(self, nodes, mask=None, _input=None, _sents=None, sent_nums=None):
        if self._type == 'gold':
            assert mask is not None
            nodes = mask.unsqueeze(2) * nodes
            return nodes, mask
        elif self._type == 'none':
            nodes = nodes
            bs, ns, fs = nodes.size()
            mask = torch.ones(bs, ns)
            return nodes, mask
        elif self._type == 'soft':
            assert _input is not None
            mask = F.sigmoid(self._mask(_input)) # B * N * 1
            nodes = nodes * mask
            return nodes, mask
        elif self._type == 'soft+sent':
            assert _input is not None and _sents is not None
            # bs, sn, ds = _sents.size()
            noden = _input.size(1)
            # sents = _sents.unsqueeze(1).repeat(1, noden, 1, 1)
            # _input = _input.unsqueeze(2).repeat(1, 1, sn, 1)
            # attention = self._bi_attn(_input, sents).squeeze(3)

            _nodes_feat = torch.matmul(_input, self._bi_attn.unsqueeze(0))
            attention = torch.matmul(_nodes_feat, _sents.permute(0, 2, 1))


            if sent_nums is not None:
                sent_mask = len_mask(sent_nums, _input.device).unsqueeze(1).repeat(1, noden, 1)
                attention = attention.masked_fill(sent_mask == 0, -1e18)
                score = F.softmax(attention, dim=-1)
                #attention = sent_mask.unsqueeze(1) * attention
            else:
                score = F.softmax(attention, dim=-1)
            weights = torch.matmul(score, _sents)
            output = torch.matmul(weights, self._wc.unsqueeze(0)) + torch.matmul(_input, self._wh.unsqueeze(0)) # B * N * emb_dim
            mask = F.sigmoid(torch.matmul(F.tanh(output), self._v.unsqueeze(0).unsqueeze(2)))
            nodes = nodes * mask
            return nodes, mask
        else:
            raise Exception('Not Implemented yet')


# class graph_sent_encode(nn.Module):
#     def __init__(self,args):
#         super().__init__()
#         self.args = args
#         rtoks = args.get('rtoks', 1)
#         hidden_size = args.get('graph_hsz', 300)
#         blockdrop = args.get('blockdrop', 0.1)
#         prop = args.get('prop', 6)
#         sparse = args.get('sparse', False)
#         model = args.get('graph_model', 'transformer')
#         self._entity = args.get('entity', False)
#
#         self.renc = nn.Embedding(rtoks, hidden_size)
#         nn.init.xavier_normal_(self.renc.weight)
#
#         if model == "gat":
#             self.gat = nn.ModuleList([MultiHeadAttention(hidden_size, hidden_size, hidden_size, h=4, dropout_p=blockdrop) for _ in range(prop)])
#         else:
#             self.gat = nn.ModuleList([Block(args) for _ in range(prop)])
#
#         self._init_dec = nn.Parameter(torch.Tensor(1, hidden_size))
#         nn.init.uniform_(self._init_dec)
#
#         if self._entity:
#             self._pad_entity = nn.Parameter(torch.Tensor(1, hidden_size))
#             nn.init.uniform_(self._pad_entity)
#
#
#         self.prop = prop
#         self.sparse = sparse
#         self.model = model
#         self.hidden_size = hidden_size
#
#     def pad(self,tensor,length):
#         return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])
#
#     def forward(self,adjs,rels,ents):
#
#         sents, sentlens, vents, entlens = ents
#         vrels = [self.renc(x) for x in rels]
#
#         glob = []
#         graphs = []
#         for i, adj in enumerate(adjs):
#             if self._entity:
#                 vgraph = torch.cat((sents[i][:sentlens[i]], self._init_dec, self._pad_entity, vents[i][:entlens[i]], vrels[i]),0)
#             else:
#                 vgraph = torch.cat((sents[i][:sentlens[i]], self._init_dec, vrels[i]), 0)
#             N = vgraph.size(0)
#             if self.sparse:
#                 lens = [len(x) for x in adj]
#                 m = max(lens)
#                 mask = torch.arange(0,m).unsqueeze(0).repeat(len(lens),1).long()
#                 mask = (mask <= torch.LongTensor(lens).unsqueeze(1)).cuda()
#                 mask = (mask == 0).unsqueeze(1)
#             else:
#                 mask = (adj == 0).unsqueeze(1)
#             for j in range(self.prop):
#                 if self.sparse:
#                     ngraph = [vgraph[k] for k in adj]
#                     ngraph = [self.pad(x,m) for x in ngraph]
#                     ngraph = torch.stack(ngraph,0)
#                     #print(ngraph.size(),vgraph.size(),mask.size())
#                     vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
#                 else:
#                     ngraph = torch.tensor(vgraph.repeat(N,1).view(N,N,-1),requires_grad=False)
#                     vgraph = self.gat[j](vgraph.unsqueeze(1),ngraph,mask)
#                     if math.isnan(vgraph.sum()):
#                         print('vgraph:', vgraph)
#                         print('mask:', mask)
#                         print('adj', adj)
#                         print(j, i)
#                     if self.model == 'gat':
#                         vgraph = vgraph.squeeze(1)
#                         vgraph = self.gatact(vgraph)
#             graphs.append(vgraph)
#             glob.append(vgraph[entlens[i]])
#         elens = [x.size(0) for x in graphs]
#         gents = [self.pad(x,max(elens)) for x in graphs]
#         gents = torch.stack(gents,0)
#         elens = torch.LongTensor(elens)
#         emask = torch.arange(0,gents.size(1)).unsqueeze(0).repeat(gents.size(0),1).long()
#         emask = (emask <= elens.unsqueeze(1)).cuda()
#         glob = torch.stack(glob,0)
#         if math.isnan(gents.sum()):
#             for gent in gents:
#                 if math.isnan(gent.sum()):
#                     print('gent:', gent)
#         return None, glob ,(gents, emask)
#
#
class subgraph_encode(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = args.get('graph_hsz', 300)
        prop = args.get('prop', 1)
        sparse = args.get('sparse', False)
        model = args.get('graph_model', 'transformer')
        adj_type = args.get('adj_type', 'edge_as_node')
        self._adj_type = adj_type

        self._rtrans = nn.Linear(hidden_size, hidden_size)

        self.gat = Block(args)
        # if model == "gat":
        #     self.gat = nn.ModuleList([MultiHeadAttention(hidden_size, hidden_size, hidden_size, h=4, dropout_p=blockdrop) for _ in range(prop)])
        # else:
        #     self.gat = nn.ModuleList([Block(args) for _ in range(prop)])

        # self._pad_entity = nn.Parameter(torch.Tensor(1, hidden_size))
        # nn.init.uniform_(self._pad_entity)

        self._pad_subgraph = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.uniform_(self._pad_subgraph)
        self._hidden = hidden_size

        assert hidden_size % 2 == 0
        self.topic_lstm = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)

        self.prop = prop
        self.sparse = sparse
        self.model = model


    def pad(self,tensor,length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(0)])


    def forward(self, batch_adjs, nodes, node_lists, output_node_rep=False):
        # adjs batch, article, subgraph
        # node lists batch, article, subgraph
        topic_length = []
        topic_reps = []
        node_align_paras = []
        node_reps = []
        node_length = []
        for _bid, adjs in enumerate(batch_adjs):
            subgraph_outs = []
            node_align_para = []
            node_outs = []
            for _sid, adj in enumerate(adjs):
                if len(adj) > 0:
                    node_list = node_lists[_bid][_sid]

                    subgraph_nodes = nodes[_bid, node_list, :]

                    mask = (adj == 0)
                    subgraph_out = self.gat(subgraph_nodes, subgraph_nodes, mask)
                    node_num, _ = subgraph_out.size()
                    node_align_para.extend([_sid for _ in range(node_num)])
                    para_rep = subgraph_out.max(dim=0)[0]
                    subgraph_outs.append(para_rep)

                    if output_node_rep:
                        node_outs.append(subgraph_out)
                else:
                    subgraph_outs.append(self._pad_subgraph.to(nodes.device))
            if output_node_rep:
                if len(node_outs) == 0:
                    node_outs.append(self._pad_subgraph.unsqueeze(0))
                    node_align_para.append(0)
                node_outs = torch.cat(node_outs, dim=0)
                node_length.append(node_outs.size(0))
                node_reps.append(node_outs)
            node_align_paras.append(node_align_para)
            length = len(subgraph_outs)
            subgraph_outs = torch.stack(subgraph_outs, dim=0)
            topic_length.append(length)
            topic_reps.append(subgraph_outs)

        # topic_reps
        max_para = max(topic_length)
        # topic_reps, topic_length = zip(*sorted(zip(topic_reps, topic_length), key=lambda x:x[1], reverse=True))
        topic_length = list(topic_length)
        topic_reps = torch.stack([torch.cat([topic_rep, torch.zeros(max_para - n, topic_rep.size(1)).to(topic_rep.device)])
                                  if n != max_para
                                  else topic_rep
                                  for topic_rep, n in zip(topic_reps, topic_length)], dim=0)
        if output_node_rep:
            max_node = max(node_length)
            node_reps = torch.stack([torch.cat([node_rep, torch.zeros(max_node - n, node_rep.size(1)).to(node_rep.device)])
                                  if n != max_node
                                  else node_rep
                                  for node_rep, n in zip(node_reps, node_length)], dim=0)


        # LSTM
        lstm_out = pack_padded_sequence(topic_reps, topic_length, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.topic_lstm(lstm_out)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        if output_node_rep:
            return (lstm_out.permute(1, 0, 2).contiguous(), topic_length), (node_reps, node_length, node_align_paras)
        else:
            return lstm_out.permute(1, 0, 2).contiguous(), topic_length


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propagator(nn.Module):
    """
    Gated Propagator for GGNN
    Using GRU gating mechanism
    """
    def __init__(self, state_dim, n_edge_types):
        super(Propagator, self).__init__()

        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )
        self.state_dim = state_dim

    def forward(self, state_in, state_out, state_cur, A): #A = [A_in, A_out]
        n_nodes = A.size(1)
        A_in = A[:, :, :n_nodes*self.n_edge_types]
        A_out = A[:, :, n_nodes*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in) #batch size x |V| x state dim
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2) #batch size x |V| x 3*state dim

        r = self.reset_gate(a.view(-1, self.state_dim*3)) #batch size*|V| x state_dim
        z = self.update_gate(a.view(-1, self.state_dim*3))
        r = r.view(-1, n_nodes, self.state_dim)
        z = z.view(-1, n_nodes, self.state_dim)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.transform(joined_input.view(-1, self.state_dim*3))
        h_hat = h_hat.view(-1, n_nodes, self.state_dim)
        output = (1 - z) * state_cur + z * h_hat
        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, n_edge_types, n_steps):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propagation Model
        self.propagator = Propagator(self.state_dim, self.n_edge_types)

        # Output Model
        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, A):
        n_nodes = prop_state.size(1)
        for i_step in range(self.n_steps):
            #print ("PROP STATE SIZE:", prop_state.size()) #batch size x |V| x state dim
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state.view(-1, self.state_dim)))
                out_states.append(self.out_fcs[i](prop_state.view(-1, self.state_dim)))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, n_nodes*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, n_nodes*self.n_edge_types, self.state_dim) #batch size x |V||E| x state dim

            prop_state = self.propagator(in_states, out_states, prop_state, A)

        #output = output.sum(1)
        return prop_state


class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(in_features, out_features))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, _input, adj):
        support = torch.matmul(_input, self.weight)
        output = torch.bmm(adj, support)

        return output + self.bias.unsqueeze(0).unsqueeze(1)


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'