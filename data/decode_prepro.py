from data.batcher import tokenize, preproc
from data.batcher import create_word_freq_in_para_feat, MAX_FREQ
from cytoolz import identity, concat, curry
from collections import Counter, defaultdict
from data.batcher import conver2id, create_sent_node_align
from utils import PAD, UNK, START, END

BERT_MAX_LEN = 512

@curry
def prepro_bert(source_sents, tokenizer, max_len, stride):

    tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in source_sents]
    tokenized_sents = [tokenized_sent + ['[SEP]'] for tokenized_sent in tokenized_sents]
    tokenized_sents[0] = ['[CLS]'] + tokenized_sents[0]
    word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
    truncated_word_num = []
    total_count = 0
    for num in word_num:
        if total_count + num < max_len:
            truncated_word_num.append(num)
        else:
            truncated_word_num.append(max_len - total_count)
            break
        total_count += num
    tokenized_sents = list(concat(tokenized_sents))[:max_len]

    tokenized_sents_lists = [tokenized_sents[:BERT_MAX_LEN]]
    length = len(tokenized_sents) - BERT_MAX_LEN
    i = 1
    while length > 0:
        tokenized_sents_lists.append(tokenized_sents[i*BERT_MAX_LEN-stride:(i+1)*BERT_MAX_LEN-stride])
        i += 1
        length -= (BERT_MAX_LEN - stride)
    id_sents = [tokenizer.convert_tokens_to_ids(tokenized_sents) for tokenized_sents in tokenized_sents_lists]
    src_num = len(truncated_word_num)
    return id_sents, truncated_word_num




@curry
def prepro_gat(tokenizer, align, batch, max_len=1024, stride=256, node_max_len=30, key='InSalientSent'):
    source_sents, nodes, edges = batch
    #original_order = ' '.join(source_sents).lower().split(' ')
    original_order = ' [sep] '.join(source_sents).lower().split(' ')
    order_match = {}
    count = 1
    i = 0
    for word in original_order:
        if word != '[sep]':
            order_match[i] = list(range(count, count + align[word]))
            count += align[word]
            i += 1
        else:
            count += 1
    tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in source_sents]
    tokenized_sents = [tokenized_sent + ['[SEP]'] for tokenized_sent in tokenized_sents]
    tokenized_sents[0] = ['[CLS]'] + tokenized_sents[0]
    word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
    truncated_word_num = []
    total_count = 0
    for num in word_num:
        if total_count + num < max_len:
            truncated_word_num.append(num)
        else:
            truncated_word_num.append(max_len - total_count)
            break
        total_count += num
    tokenized_sents = list(concat(tokenized_sents))[:max_len]
    tokenized_sents_lists = [tokenized_sents[:BERT_MAX_LEN]]
    length = len(tokenized_sents) - BERT_MAX_LEN
    i = 1
    while length > 0:
        tokenized_sents_lists.append(tokenized_sents[i * BERT_MAX_LEN - stride:(i + 1) * BERT_MAX_LEN - stride])
        i += 1
        length -= (BERT_MAX_LEN - stride)
    # find out of range and useless nodes
    other_nodes = set()
    oor_nodes = []  # out of range nodes will not included in the graph
    for _id, content in nodes.items():
        words = [_ for mention in content['content:'] for pos in mention['word_pos'] if
                 pos != -1 and order_match.__contains__(pos) for _ in
                 order_match[pos]]
        words = [word for word in words if word < max_len]
        if len(words) != 0:
            other_nodes.add(_id)
        else:
            oor_nodes.append(_id)
    activated_nodes = set()
    for _id, content in edges.items():
        if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
            words = content['content']['word_pos']
            new_words = [_ for word in words if word > -1 and order_match.__contains__(word) for _ in
                         order_match[word] if _ < max_len]
            if len(new_words) > 0:
                activated_nodes.add(content['content']['arg1'])
                activated_nodes.add(content['content']['arg2'])
    oor_nodes.extend(list(other_nodes - activated_nodes))
    # process nodes
    sorted_nodes = sorted(nodes.items(), key=lambda x: int(x[0].split('_')[1]))
    nodewords = []
    sum_worthy = []
    id2node = {}
    ii = 0
    for _id, content in sorted_nodes:
        if _id not in oor_nodes:
            words = [_ for mention in content['content:'] for pos in mention['word_pos'] if
                     pos != -1 and order_match.__contains__(pos) for _ in order_match[pos]]
            words = [word for word in words if word < max_len]
            words = words[:node_max_len]
            sum_worthy.append(content[key])
            if len(words) != 0:
                nodewords.append(words)
                id2node[_id] = ii
                ii += 1
            else:
                oor_nodes.append(_id)
    if len(nodewords) == 0:
        # print('warning! no nodes in this sample')
        nodewords = [[0], [2]]
        sum_worthy.extend([0, 0])
    nodelength = [len(words) for words in nodewords]
    # process edges
    acticated_nodes = set()
    triples = []
    relations = []
    sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))
    ii = 0
    for _id, content in sorted_edges:
        if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
            words = content['content']['word_pos']
            new_words = [_ for word in words if word > -1 and order_match.__contains__(word) for _ in
                         order_match[word] if _ < max_len]
            new_words = new_words[:node_max_len]
            if len(new_words) > 0:
                node1 = id2node[content['content']['arg1']]
                node2 = id2node[content['content']['arg2']]
                edge = int(_id.split('_')[1])
                triples.append([node1, ii, node2])
                acticated_nodes.add(content['content']['arg1'])
                acticated_nodes.add(content['content']['arg2'])
                ii += 1
                relations.append(new_words)
    if len(relations) == 0:
        # print('warning! no edges in this sample')
        relations = [[1]]
        triples = [[0, 0, 1]]
    rlength = [len(words) for words in relations]
    id_sents = [tokenizer.convert_tokens_to_ids(tokenized_sents) for tokenized_sents in tokenized_sents_lists]

    return id_sents, truncated_word_num, (nodewords, nodelength, sum_worthy), (relations, rlength, triples)


@curry
def prepro_gat_nobert(batch, max_sent_len=100, max_sent=60, node_max_len=30, key='summary_worthy', adj_type='concat_triple'):
    source_sents, nodes, edges, paras = batch
    tokenized_sents = tokenize(max_sent_len, source_sents)[:max_sent]
    tokenized_sents_2 = tokenize(None, source_sents)[:max_sent]
    tokenized_article = list(concat(tokenized_sents_2))
    max_len = len(tokenized_article)
    # tokenized_sents = [tokenized_sent + ['[SEP]'] for tokenized_sent in tokenized_sents]
    # tokenized_sents[0] = ['[CLS]'] + tokenized_sents[0]
    word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
    truncated_word_num = word_num
    # find out of range and useless nodes
    other_nodes = set()
    oor_nodes = []  # out of range nodes will not included in the graph
    word_freq_feat, word_inpara_feat, sent_freq_feat, sent_inpara_freq_feat  = create_word_freq_in_para_feat(paras, tokenized_sents, tokenized_article)
    assert len(word_freq_feat) == len(tokenized_article) and len(word_inpara_feat) == len(tokenized_article)
    for _id, content in nodes.items():

        words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
        words = [word for word in words if word < max_len]
        if len(words) != 0:
            other_nodes.add(_id)
        else:
            oor_nodes.append(_id)
    activated_nodes = set()
    for _id, content in edges.items():
        if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
            words = content['content']['word_pos']
            new_words = [word for word in words if word > -1 and word < max_len]
            if len(new_words) > 0:
                activated_nodes.add(content['content']['arg1'])
                activated_nodes.add(content['content']['arg2'])
    oor_nodes.extend(list(other_nodes - activated_nodes))
    # process nodes
    sorted_nodes = sorted(nodes.items(), key=lambda x: int(x[0].split('_')[1]))
    nodewords = []
    nodefreq = []
    nodeinsent = []
    nodetype = []
    sum_worthy = []
    id2node = {}
    ii = 0
    for _id, content in sorted_nodes:
        if _id not in oor_nodes:
            words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
            words = [word for word in words if word < max_len]
            words = words[:node_max_len]
            sum_worthy.append(content[key])
            if len(words) != 0:
                nodewords.append(words)
                nodefreq.append(len(content['content']))
                nodetype.append(1)
                nodeinsent.append([mention['sent_pos'] for mention in content['content'] if
                                   mention['sent_pos'] < len(tokenized_sents)])
                id2node[_id] = ii
                ii += 1
            else:
                oor_nodes.append(_id)
    if len(nodewords) == 0:
        # print('warning! no nodes in this sample')
        nodewords = [[0], [2]]
        nodefreq.extend([1, 1])
        nodeinsent.extend([[0], [0]])
        nodetype.extend([1, 1])
        sum_worthy.extend([0, 0])
    nodelength = [len(words) for words in nodewords]
    # process edges
    acticated_nodes = set()
    triples = []
    edge_freq = []
    edgeinsent = []
    edgetype = []
    relations = []
    sum_worthy_edges = []
    sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))
    ii = 0
    for _id, content in sorted_edges:
        if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
            words = content['content']['word_pos']
            new_words = [word for word in words if word > -1 and word < max_len]
            new_words = new_words[:node_max_len]
            if len(new_words) > 0:
                node1 = id2node[content['content']['arg1']]
                node2 = id2node[content['content']['arg2']]
                sum_worthy_edges.append(content[key])
                try:
                    sent_pos = [content['content']['sent_pos']]
                except KeyError:
                    sent_pos = [content['content']['arg1_original'][0]['sent_pos']]
                if adj_type == 'edge_up':
                    nodewords[node1].extend(new_words)
                elif adj_type == 'edge_down':
                    nodewords[node2].extend(new_words)
                edge = int(_id.split('_')[1])
                edge_freq.append(1)
                edgeinsent.append(sent_pos)
                edgetype.append(2)
                triples.append([node1, ii, node2])
                acticated_nodes.add(content['content']['arg1'])
                acticated_nodes.add(content['content']['arg2'])
                ii += 1
                relations.append(new_words)
    if len(relations) == 0:
        # print('warning! no edges in this sample')
        relations = [[1]]
        triples = [[0, 0, 1]]
        edgeinsent.append([0])
        edge_freq = [1]
        edgetype.append(2)
        sum_worthy_edges.extend([0])
    nodefreq = [freq if freq < MAX_FREQ - 1 else MAX_FREQ - 1 for freq in nodefreq]
    rlength = [len(words) for words in relations]
    if adj_type == 'edge_as_node':
        nodewords = nodewords + relations
        nodelength = nodelength + rlength
        sum_worthy = sum_worthy + sum_worthy_edges
        nodefreq = nodefreq + edge_freq
        nodetype = nodetype + edgetype
        nodeinsent = nodeinsent + edgeinsent

    sent_node_aligns = create_sent_node_align(nodeinsent, len(tokenized_sents))

    return tokenized_article, truncated_word_num, (nodewords, nodelength, sum_worthy, nodefreq, word_freq_feat, word_inpara_feat, sent_freq_feat, sent_inpara_freq_feat, sent_node_aligns), \
           (relations, rlength, triples)


@curry
def prepro_subgraph_nobert(batch, max_sent_len=100, max_sent=60, node_max_len=30, key='InSalientSent', adj_type='edge_as_node'):
    source_sents, nodes, edges, subgraphs, paras, extracts  = batch
    tokenized_sents = tokenize(max_sent_len, source_sents)[:max_sent]
    tokenized_sents_2 = tokenize(None, source_sents)[:max_sent]
    tokenized_article = list(concat(tokenized_sents_2))
    cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                   extracts))
    max_len = len(tokenized_article)
    # tokenized_sents = [tokenized_sent + ['[SEP]'] for tokenized_sent in tokenized_sents]
    # tokenized_sents[0] = ['[CLS]'] + tokenized_sents[0]

    sent_align_para = []
    last_idx = 0
    for sent in range(len(tokenized_sents)):
        flag = False
        for _idx, para in enumerate(paras):
            if sent in para:
                sent_align_para.append(_idx)
                last_idx = _idx
                flag = True
                break
        if not flag:
            sent_align_para.append(last_idx)
    assert len(sent_align_para) == len(tokenized_sents)
    sent_align_para.append(last_idx + 1)
    segment_feat_para = [sent_align_para[_sid] + 1 if sent_align_para[_sid] < MAX_FREQ-1 else MAX_FREQ-1 for _sid, sent in enumerate(tokenized_sents_2) for word in sent]
    segment_feat_sent = [[sent_align_para[_sid] + 1 if sent_align_para[_sid] < MAX_FREQ-1 else MAX_FREQ-1 for word in sent] for _sid, sent in enumerate(tokenized_sents)]

    sent_align_para = [[_] for _ in sent_align_para]

    word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
    truncated_word_num = word_num
    # find out of range and useless nodes
    other_nodes = set()
    oor_nodes = []  # out of range nodes will not included in the graph
    word_freq_feat, word_inpara_feat, sent_freq_feat, sent_inpara_freq_feat = create_word_freq_in_para_feat(paras, tokenized_sents, tokenized_article)
    assert len(word_freq_feat) == len(tokenized_article) and len(word_inpara_feat) == len(tokenized_article)

    for _id, content in nodes.items():

        words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
        words = [word for word in words if word < max_len]
        if len(words) != 0:
            other_nodes.add(_id)
        else:
            oor_nodes.append(_id)
    activated_nodes = set()
    for _id, content in edges.items():
        if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
            words = content['content']['word_pos']
            new_words = [word for word in words if word > -1 and word < max_len]
            if len(new_words) > 0:
                activated_nodes.add(content['content']['arg1'])
                activated_nodes.add(content['content']['arg2'])
    oor_nodes.extend(list(other_nodes - activated_nodes))
    # process nodes
    sorted_nodes = sorted(nodes.items(), key=lambda x: int(x[0].split('_')[1]))
    nodewords = []
    nodefreq = []
    sum_worthy = []
    id2node = {}
    ii = 0
    for _id, content in sorted_nodes:
        if _id not in oor_nodes:
            words = [pos for mention in content['content'] for pos in mention['word_pos'] if pos != -1]
            words = [word for word in words if word < max_len]
            words = words[:node_max_len]
            sum_worthy.append(content[key])
            if len(words) != 0:
                nodewords.append(words)
                nodefreq.append(len(content['content']))
                id2node[_id] = ii
                ii += 1
            else:
                oor_nodes.append(_id)
    if len(nodewords) == 0:
        # print('warning! no nodes in this sample')
        nodewords = [[0], [2]]
        nodefreq.extend([1, 1])
        sum_worthy.extend([0, 0])
    nodelength = [len(words) for words in nodewords]
    # process edges
    acticated_nodes = set()
    triples = []
    edge_freq = []
    relations = []
    sum_worthy_edges = []
    id2edge = {}
    sorted_edges = sorted(edges.items(), key=lambda x: int(x[0].split('_')[1]))
    ii = 0
    for _id, content in sorted_edges:
        if content['content']['arg1'] not in oor_nodes and content['content']['arg2'] not in oor_nodes:
            words = content['content']['word_pos']
            new_words = [word for word in words if word > -1 and word < max_len]
            new_words = new_words[:node_max_len]
            if len(new_words) > 0:
                node1 = id2node[content['content']['arg1']]
                node2 = id2node[content['content']['arg2']]
                sum_worthy_edges.append(content[key])
                if adj_type == 'edge_up':
                    nodewords[node1].extend(new_words)
                elif adj_type == 'edge_down':
                    nodewords[node2].extend(new_words)
                edge = int(_id.split('_')[1])
                edge_freq.append(1)
                triples.append([node1, ii, node2])
                acticated_nodes.add(content['content']['arg1'])
                acticated_nodes.add(content['content']['arg2'])
                id2edge[_id] = ii
                ii += 1
                relations.append(new_words)
    if len(relations) == 0:
        # print('warning! no edges in this sample')
        relations = [[1]]
        edge_freq = [1]
        triples = [[0, 0, 1]]
        sum_worthy_edges.extend([0])

    node_lists = []
    edge_lists = []
    triples = []
    if max_sent is None:
        max_sent = 9999

    for _sgid, subgraph in enumerate(subgraphs):
        try:
            paraid = paras[_sgid][0]
        except:
            paraid = 0
        if type(paraid) != type(max_sent):
            paraid = 0
        if paraid > max_sent - 1:
            continue
        if subgraph == []:
            node_lists.append([])
            triples.append([])
            edge_lists.append([])
        else:
            node_list = set()
            triple = []
            edge_list = []
            eidx = []
            for _triple in subgraph:
                if _triple[0] not in oor_nodes and _triple[2] not in oor_nodes and id2edge.__contains__(_triple[1]):
                    node_list.add(id2node[_triple[0]])
                    node_list.add(id2node[_triple[2]])
                    eidx.append(_triple[1])
            node_list = list(sorted(node_list))
            for _triple in subgraph:
                if _triple[0] not in oor_nodes and _triple[2] not in oor_nodes and id2edge.__contains__(_triple[1]):
                    idx1 = node_list.index(id2node[_triple[0]])
                    idx2 = node_list.index(id2node[_triple[2]])
                    _idxe = id2edge[_triple[1]]
                    idxe_in_subgraph = eidx.index(_triple[1])
                    edge_list.append(_idxe)
                    triple.append([idx1, idxe_in_subgraph, idx2])
            triples.append(triple)
            node_lists.append(node_list)
            edge_lists.append(edge_list)

    if len(node_lists) == 0:
        node_lists.append([])
        triples.append([])
        edge_lists.append([])


    rlength = [len(words) for words in relations]
    nodefreq = [freq if freq < MAX_FREQ - 1 else MAX_FREQ - 1 for freq in nodefreq]
    if adj_type == 'edge_as_node':
        node_num = len(nodewords)
        nodewords = nodewords + relations
        nodefreq = nodefreq + edge_freq
        nodelength = nodelength + rlength
        sum_worthy = sum_worthy + sum_worthy_edges
        for i in range(len(triples)):
            node_lists[i] = node_lists[i] + [edge + node_num for edge in edge_lists[i]]

    gold_dec_selection_label = [0 for i in range(len(node_lists))]
    for sent in cleaned_extracts:
        for i, para in enumerate(paras):
            if sent in para:
                gold_dec_selection_label[i] = 1

    return tokenized_article, truncated_word_num, (nodewords, sum_worthy, gold_dec_selection_label), (relations, triples, node_lists,
            sent_align_para, segment_feat_sent, segment_feat_para, nodefreq, word_freq_feat, word_inpara_feat, sent_freq_feat, sent_inpara_freq_feat)