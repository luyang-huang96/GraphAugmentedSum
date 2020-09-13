import json, os
import multiprocessing as mp
import time, glob
from cytoolz import curry
from datetime import timedelta
import gc
import os
from nltk.stem import porter


DATA_DIR = ''
OPENIE_DIR = ''
WRITE_DIR = ''

def read_raw(split):
    data_dict = {}
    with open(os.path.join(OPENIE_DIR, 'cnndm_' + split + '_out.txt'), 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            data = [_.strip() for _ in data]
            try:
                data_dict[data[0]][data[1]].append(data[2:])
            except KeyError:
                try:
                    data_dict[data[0]].setdefault(data[1], [data[2:]])
                except:
                    data_dict[data[0]] = {data[1]: [data[2:]]}
    return data_dict

def read_raw_2(split):
    data_dict = {}
    with open(os.path.join(OPENIE_DIR, split + '_out.txt'), 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            data = [_.strip() for _ in data]
            data_dict.setdefault(data[0], [])

    return data_dict




def prepare_data(filename, split='train'):
    with open(filename, 'r') as g:
        data = json.load(g)
    coref = data['input_mention_cluster']
    abstract = data['abstract']
    article = data['article']
    if split != 'test':
        extracts = data['extracted_combine']
    else:
        extracts = []
    abstract = ' '.join(abstract)
    id2node = {}
    _id = 0
    for entity in coref:
        entities = []
        for mention in entity:
            entities.append((mention['text'], mention['position']))
        _id += 1
        id2node.setdefault('entity_' + str(_id), entities)

    return abstract, id2node, data, article, extracts

def filter_arguments(f_arguments, t_arguments, s_arguments):
    if len(f_arguments) > 1:
        f_arguments = list(f_arguments)
        f_arguments.sort(key=lambda x: len(x[0]), reverse=True)
        start = int(f_arguments[0][2])
        end = int(f_arguments[0][3])
        new_f_arguments = [f_arguments[0]]
        for candidate in f_arguments:
            if ((int(candidate[2]) > start - 1) and (int(candidate[3]) < end + 1)) or (start > int(candidate[2]) - 1 and end < int(candidate[3]) + 1):
                continue
            else:
                new_f_arguments.append(candidate)
        f_arguments = new_f_arguments
    if len(t_arguments) > 1:
        t_arguments = list(t_arguments)
        t_arguments.sort(key=lambda x: len(x[0]), reverse=True)
        start = int(t_arguments[0][1])
        end = int(t_arguments[0][2])
        new_t_arguments = [t_arguments[0]]
        for candidate in t_arguments:
            if (int(candidate[1]) > start - 2 and int(candidate[2]) < end + 2) or (start > int(candidate[1]) - 1 and end < int(candidate[2]) + 1):
                continue
            else:
                new_t_arguments.append(candidate)
        t_arguments = new_t_arguments
    if len(s_arguments) > 1:
        s_arguments = list(s_arguments)
        s_arguments.sort(key=lambda x: len(x[0]), reverse=True)
        start = int(s_arguments[0][2])
        end = int(s_arguments[0][3])
        new_s_arguments = [s_arguments[0]]
        for candidate in s_arguments:
            if (int(candidate[2]) > start - 2 and int(candidate[3]) < end + 2) or (start > int(candidate[2]) - 1 and end < int(candidate[3]) + 1):
                continue
            else:
                new_s_arguments.append(candidate)
        s_arguments = new_s_arguments
    return f_arguments, t_arguments, s_arguments


def merge(clusters):
    new_cluster = []
    length = len(clusters)
    not_finished = list(range(length))
    while(len(not_finished) > 0):
        target = not_finished[0]
        not_finished.remove(target)
        target_cluster = clusters[target]
        new_not_finished = []
        for _id in not_finished:
            cluster = clusters[_id]
            if len(set(cluster) & set(target_cluster)) > 0:
                target_cluster += cluster
            else:
                new_not_finished.append(_id)
        not_finished = new_not_finished
        new_cluster.append(set(target_cluster))
    merged = new_cluster
    return merged

def find_share_mention(tuples): # rule 2
    sentd_ids = set(list(range(len(tuples))))
    # find aligned tuples
    clusters = []
    for id, tuple in enumerate(tuples):
        for _id, _tuple in enumerate(tuples):
            if _id < id + 1:
                continue
            if (_tuple[0] == tuple[0] and _tuple[1] == tuple[1]) or (
                    _tuple[0] == tuple[0] and _tuple[2] == tuple[2]) or (
                    _tuple[2] == tuple[2] and _tuple[1] == tuple[1]):
                clusters.append((id, _id))
    final_clusters = merge(clusters)

    for cluster in final_clusters:
        sentd_ids = sentd_ids - cluster

    final_clusters = [list(final_cluster) for final_cluster in final_clusters]
    no_merge_cluster = [[sentd_id] for sentd_id in sentd_ids]
    final_sent_clusters = no_merge_cluster + final_clusters
    return final_sent_clusters


def filter_argument(tuples, thresold=10): # rule0
    new_tuples = []
    for tuple in tuples:
        if int(tuple[4]) - int(tuple[3]) < thresold:
            if int(tuple[6]) - int(tuple[5]) < thresold:
                if int(tuple[8]) - int(tuple[7]) < thresold:
                    new_tuples.append(tuple)
    return new_tuples

def filter_tuples(final_sent_clusters, tuples, entities, sent_id): # rule 3
    arguments_list = []
    for cluster in final_sent_clusters:
        f_arguments = set()
        t_arguments = set()
        s_arguments = set()
        for i in range(len(cluster)):
            flag_1 = 0
            canonical_mention_1 = ' '
            for _id, mention_cluster in entities.items():
                for mention, _sent_id in mention_cluster:
                    _sent_id = _sent_id[0]
                    if int(_sent_id) == int(sent_id):
                        if tuples[cluster[i]][0] in mention:
                            canonical_mention_1 = _id
                            flag_1 = 1
                            break

            f_arguments.add((tuples[cluster[i]][0], canonical_mention_1, int(tuples[cluster[i]][3]), int(tuples[cluster[i]][4])))

            t_arguments.add((tuples[cluster[i]][1], int(tuples[cluster[i]][5]), int(tuples[cluster[i]][6])))

            canonical_mention_2 = ' '
            flag_2 = 0
            for _id, mention_cluster in entities.items():
                for mention, _sent_id in mention_cluster:
                    _sent_id = _sent_id[0]
                    if int(_sent_id) == int(sent_id):
                        if tuples[cluster[i]][2] in mention:
                            canonical_mention_2 = _id
                            flag_2 = 1
                            break
            if (flag_1 and flag_2) and canonical_mention_1 == canonical_mention_2:
                s_arguments.add((tuples[cluster[i]][2], ' ', int(tuples[cluster[i]][7]), int(tuples[cluster[i]][8])))
            else:
                s_arguments.add(
                    (tuples[cluster[i]][2], canonical_mention_2, int(tuples[cluster[i]][7]), int(tuples[cluster[i]][8])))

        # align arguments
        unfinished = []
        if len(f_arguments) + len(t_arguments) + len(s_arguments) > 3:
            unfinished.append((list(f_arguments), list(t_arguments), list(s_arguments)))
        else:
            arguments_list.append((list(f_arguments)[0], list(t_arguments)[0], list(s_arguments)[0]))
        while len(unfinished) > 0:
            f_arguments, t_arguments, s_arguments = unfinished.pop()
            f_arguments, t_arguments, s_arguments = filter_arguments(f_arguments, t_arguments, s_arguments)
            if len(f_arguments) > 1:
                unfinished.append((f_arguments[1:], t_arguments, s_arguments))
                f_arguments, t_arguments, s_arguments = [f_arguments[0]], t_arguments, s_arguments
            if len(t_arguments) > 1:
                unfinished.append((f_arguments, t_arguments[1:], s_arguments))
                f_arguments, t_arguments, s_arguments = f_arguments, [t_arguments[0]], s_arguments
            if len(s_arguments) > 1:
                unfinished.append((f_arguments, t_arguments, s_arguments[1:]))
                f_arguments, t_arguments, s_arguments = f_arguments, t_arguments, [s_arguments[0]]
            arguments_list.append((f_arguments[0], t_arguments[0], s_arguments[0]))
    return arguments_list

def filter_stopwords(arguments, stopwords):
    def filter_one(words):
        return all([word.strip() in stopwords for word in words])
    new_arguments = []
    for argument in arguments:
        flags = [filter_one(argument[0][1].lower().split(' ') + argument[0][0].lower().split(' ')),
                 filter_one(argument[1][0].lower().split(' ')), filter_one(argument[2][1].lower().split(' ')  + argument[2][0].lower().split(' '))]
        if not all(flags):
            new_arguments.append(argument)
    return new_arguments

def get_stopwods():
    stopwords = []
    with open('/home/luyang/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords

def final_process(all_arguments, entities, article):
    finished_entity = set()
    entity2node = {}
    nodes = {}
    edges = {}
    word_nums = [len(sent.split(' ')) for sent in article]
    _id = 0
    _id_edge = 0
    for sent_id, arguments in all_arguments:
        sent = article[sent_id].split()
        start_num = sum(word_nums[:sent_id])

        for argument in arguments:
            e1 = argument[0]
            r = argument[1]
            e2 = argument[2]
            # process e1
            if e1[1] != ' ':
                if e1[1] not in finished_entity:
                    entity = entities[e1[1]]
                    entity_processed = []
                    for mention in entity:
                        info = {}
                        info['text'] = mention[0]
                        info['word_pos'] = list(range(int(mention[1][1]), int(mention[1][2])))
                        info['insent_pos'] = list(range(int(mention[1][3]), int(mention[1][4])))
                        info['sent_pos'] = int(mention[1][0])
                        entity_processed.append(info)

                    e1_node = 'node_' + str(_id)
                    entity2node.setdefault(e1[1], e1_node)
                    nodes.setdefault('node_' + str(_id), {'content': entity_processed,
                                                          'type': 'entity'})
                    _id += 1
                    finished_entity.add(e1[1])
                else:
                    e1_node = entity2node[e1[1]]

                # save original form
                info = {}
                info['text'] = e1[0]
                _start = int(e1[2])
                _end = int(e1[3])
                info['insent_pos'] = []
                for word in e1[0].split():
                    poses = [i for i, _word in enumerate(sent) if _word == word]
                    if len(poses) > 1:
                        poses = [i for i in poses if i > _start-1 and i < _end]
                    if len(poses) < 1:
                        poses.append(-1)
                    info['insent_pos'].append(poses[0])
                info['word_pos'] = [pos + start_num for pos in info['insent_pos']]
                info['sent_pos'] = sent_id
                original_e1 = [info]
            else:
                info = {}
                info['text'] = e1[0]
                _start = int(e1[2])
                _end = int(e1[3])
                info['insent_pos'] = []
                for word in e1[0].split():
                    poses = [i for i, _word in enumerate(sent) if _word == word]
                    if len(poses) > 1:
                        poses = [i for i in poses if i > _start-1 and i < _end]
                    if len(poses) < 1:
                        poses.append(-1)
                    info['insent_pos'].append(poses[0])
                info['word_pos'] = [pos+start_num if pos > -1 else -1 for pos in info['insent_pos']]
                info['sent_pos'] = sent_id
                e1_node = 'node_' + str(_id)
                nodes.setdefault('node_' + str(_id), {'content': [info],
                                                      'type': 'other'})
                _id += 1
                original_e1 = [info]
            # process e2
            if e2[1] != ' ':
                if e2[1] not in finished_entity:
                    entity = entities[e2[1]]
                    entity_processed = []
                    for mention in entity:
                        info = {}
                        info['text'] = mention[0]
                        info['word_pos'] = list(range(int(mention[1][1]), int(mention[1][2])))
                        info['insent_pos'] = list(range(int(mention[1][3]), int(mention[1][4])))
                        info['sent_pos'] = int(mention[1][0])
                        entity_processed.append(info)

                    e2_node = 'node_' + str(_id)
                    nodes.setdefault('node_' + str(_id), {'content': entity_processed,
                                                          'type': 'entity'})
                    entity2node.setdefault(e2[1], e2_node)
                    _id += 1
                    finished_entity.add(e2[1])
                else:
                    e2_node = entity2node[e2[1]]
                # save original form
                info = {}
                info['text'] = e2[0]
                _start = int(e2[2])
                _end = int(e2[3])
                info['insent_pos'] = []
                for word in e2[0].split():
                    poses = [i for i, _word in enumerate(sent) if _word == word]
                    if len(poses) > 1:
                        poses = [i for i in poses if i > _start-1 and i < _end]
                    if len(poses) < 1:
                        poses.append(-1)
                    info['insent_pos'].append(poses[0])
                info['word_pos'] = [pos + start_num if pos > -1 else -1 for pos in info['insent_pos']]
                info['sent_pos'] = sent_id
                original_e2 = [info]
            else:
                info = {}
                info['text'] = e2[0]
                _start = int(e2[2])
                _end = int(e2[3])
                info['insent_pos'] = []
                for word in e2[0].split():
                    poses = [i for i, _word in enumerate(sent) if _word == word]
                    if len(poses) > 1:
                        poses = [i for i in poses if i > _start-1 and i < _end]
                    if len(poses) < 1:
                        poses.append(-1)
                    info['insent_pos'].append(poses[0])
                info['word_pos'] = [pos+start_num for pos in info['insent_pos']]
                info['sent_pos'] = sent_id
                e2_node = 'node_' + str(_id)
                nodes.setdefault('node_' + str(_id), {'content': [info],
                                                      'type': 'other'})
                _id += 1
                original_e2 = [info]
            # process r
            info = {}
            info['text'] = r[0]
            _start = int(r[1])
            _end = int(r[2])
            info['insent_pos'] = []
            for word in r[0].split():
                poses = [i for i, _word in enumerate(sent) if _word == word and i > _start-1 and i < _end]
                if len(poses) < 1:
                    poses.append(-1)
                info['insent_pos'].append(poses[0])
            info['word_pos'] = [pos+start_num if pos > -1 else -1 for pos in info['insent_pos']]
            info['sent_pos'] = sent_id
            info['arg1'] = e1_node
            info['arg2'] = e2_node
            info['arg1_original'] = original_e1
            info['arg2_original'] = original_e2
            edges.setdefault('edge_' + str(_id_edge), {'content':info,
                                                       'type': None})
            _id_edge += 1

    return nodes, edges


def pruning(all_arguments, thresold=3): # prune small subgraphs
    argument2triple = {}
    triples = []
    finished_entity = set()
    entity2node = {}
    nodes = {}
    edges = {}
    _id = 0
    _id_edge = 0
    for sent_id, arguments in all_arguments:
        for argument in arguments:
            triple = []
            e1 = argument[0]
            r = argument[1]
            e2 = argument[2]
            # process e1
            if e1[1] != ' ':
                if e1[1] not in finished_entity:
                    e1_node = 'node_' + str(_id)
                    triple.append(_id)
                    entity2node[e1[1]] = e1_node
                    _id += 1
                    finished_entity.add(e1[1])
                else:
                    e1_node = entity2node[e1[1]]
                    triple.append(int(e1_node.split('_')[-1]))
            else:
                e1_node = 'node_' + str(_id)
                triple.append(_id)
                _id += 1
            # process e2
            if e2[1] != ' ':
                if e2[1] not in finished_entity:
                    e2_node = 'node_' + str(_id)
                    entity2node.setdefault(e2[1], e2_node)
                    triple.append(_id)
                    _id += 1
                    finished_entity.add(e2[1])
                else:
                    e2_node = entity2node[e2[1]]
                    triple.append(int(e2_node.split('_')[-1]))
            else:
                e2_node = 'node_' + str(_id)
                triple.append(_id)
                _id += 1

            triple.append('edge_' + str(_id_edge))
            _id_edge += 1
            triples.append(triple)
            argument2triple.setdefault(argument, triple)
    numofnode = _id
    graph = [[0 for _ in range(numofnode)] for _ in range(numofnode)]
    for triple in triples:
        graph[triple[0]][triple[1]] = 1
        graph[triple[1]][triple[0]] = 1

    def subgraph_count(graph):
        size = len(graph)
        _map = set()
        unfinished = [i for i in range(len(graph))]
        queue = [0]  # start from 0
        count = 0
        groups = []
        single_node = 0
        while len(unfinished) > 0:
            if len(queue) == 0:
                groups.append(_map)
                if len(_map) == 1:
                    single_node += 1
                _map = set()
                queue.append(unfinished[0])
                count += 1
            _next = queue.pop()
            if len(_map) == 0:
                _map.add(_next)
                unfinished.remove(_next)
            for i in range(size):
                if i in unfinished and graph[_next][i] > 0:
                    _map.add(i)
                    unfinished.remove(i)
                    queue.append(i)

        if len(_map) > 0:
            groups.append(_map)
            count += 1
            groups = sorted(groups, key=lambda x: len(x), reverse=True)
        if len(groups) == 0:
            return count, None, single_node, groups
        else:
            return count, groups[0], single_node, groups

    count, lgroup, single, groups = subgraph_count(graph)
    banned_nodes = []
    for group in groups:
        if len(group) < thresold:
            for _iii in group:
                banned_nodes.append(_iii)

    new_all_arguments = []
    for sent_id, arguments in all_arguments:
        args = []
        for argument in arguments:
            if argument2triple[argument][0] not in banned_nodes:
                args.append(argument)
        new_all_arguments.append([sent_id, args])
    if len(new_all_arguments) == 0:
        new_all_arguments = all_arguments

    return new_all_arguments

def select_largest_group(all_arguments): # prune small subgraphs
    argument2triple = {}
    triples = []
    finished_entity = set()
    entity2node = {}
    nodes = {}
    edges = {}
    _id = 0
    _id_edge = 0
    for sent_id, arguments in all_arguments:
        for argument in arguments:
            triple = []
            e1 = argument[0]
            r = argument[1]
            e2 = argument[2]
            # process e1
            if e1[1] != ' ':
                if e1[1] not in finished_entity:
                    e1_node = 'node_' + str(_id)
                    triple.append(_id)
                    entity2node[e1[1]] = e1_node
                    _id += 1
                    finished_entity.add(e1[1])
                else:
                    e1_node = entity2node[e1[1]]
                    triple.append(int(e1_node.split('_')[-1]))
            else:
                e1_node = 'node_' + str(_id)
                triple.append(_id)
                _id += 1
            # process e2
            if e2[1] != ' ':
                if e2[1] not in finished_entity:
                    e2_node = 'node_' + str(_id)
                    entity2node.setdefault(e2[1], e2_node)
                    triple.append(_id)
                    _id += 1
                    finished_entity.add(e2[1])
                else:
                    e2_node = entity2node[e2[1]]
                    triple.append(int(e2_node.split('_')[-1]))
            else:
                e2_node = 'node_' + str(_id)
                triple.append(_id)
                _id += 1

            triple.append('edge_' + str(_id_edge))
            _id_edge += 1
            triples.append(triple)
            argument2triple.setdefault(argument, triple)
    numofnode = _id
    graph = [[0 for _ in range(numofnode)] for _ in range(numofnode)]
    for triple in triples:
        graph[triple[0]][triple[1]] = 1
        graph[triple[1]][triple[0]] = 1

    def subgraph_count(graph):
        size = len(graph)
        _map = set()
        unfinished = [i for i in range(len(graph))]
        queue = [0]  # start from 0
        count = 0
        groups = []
        single_node = 0
        while len(unfinished) > 0:
            if len(queue) == 0:
                groups.append(_map)
                if len(_map) == 1:
                    single_node += 1
                _map = set()
                queue.append(unfinished[0])
                count += 1
            _next = queue.pop()
            if len(_map) == 0:
                _map.add(_next)
                unfinished.remove(_next)
            for i in range(size):
                if i in unfinished and graph[_next][i] > 0:
                    _map.add(i)
                    unfinished.remove(i)
                    queue.append(i)

        if len(_map) > 0:
            groups.append(_map)
            count += 1
            groups = sorted(groups, key=lambda x: len(x), reverse=True)
        if len(groups) == 0:
            return count, None, single_node, groups
        else:
            return count, groups[0], single_node, groups

    count, lgroup, single, groups = subgraph_count(graph)
    banned_nodes = []
    for group in groups[1:]:
        for _iii in group:
            banned_nodes.append(_iii)

    new_all_arguments = []
    count = 0
    for sent_id, arguments in all_arguments:
        args = []
        for argument in arguments:
            if argument2triple[argument][0] not in banned_nodes:
                args.append(argument)
                count += 1
        if len(args) != 0:
            new_all_arguments.append([sent_id, args])
    if len(new_all_arguments) == 0:
        new_all_arguments = all_arguments

    return new_all_arguments


def make_summary_worth_node(nodes, abstract, stopwords, extracts, stemmer):
    sum_worthy = 0
    insalientsent = 0
    abstract = abstract.lower().split(' ')
    abstract = [stemmer.stem(word.lower()) for word in abstract]
    new_nodes = {}
    for _id, node in nodes.items():
        words = []
        node['summary_worthy'] = 0
        node['InSalientSent'] = 0
        for info in node['content']:
            words.extend(info['text'].lower().split(' '))
            sent = info['sent_pos']
            if sent in extracts:
                node['InSalientSent'] = 1

        for word in words:
            if word not in stopwords and word in abstract:
                node['summary_worthy'] = 1
                break
        if node['InSalientSent']:
            insalientsent += 1
        if node['summary_worthy']:
            sum_worthy += 1
        new_nodes.setdefault(_id, node)

    return new_nodes, insalientsent, sum_worthy

def make_summary_worth_edge(edges, abstract, stopwords, extracts, stemmer):
    sum_worthy = 0
    insalientsent = 0
    abstract = abstract.lower().split(' ')
    abstract = [stemmer.stem(word.lower()) for word in abstract]
    new_edges = {}
    for _id, edge in edges.items():
        words = []
        edge['summary_worthy'] = 0
        edge['InSalientSent'] = 0
        info = edge['content']
        words.extend(info['text'].lower().split(' '))
        sent = info['sent_pos']
        if sent in extracts:
            edge['InSalientSent'] = 1

        for word in words:
            if word not in stopwords and word in abstract:
                edge['summary_worthy'] = 1
                break
        if edge['InSalientSent']:
            insalientsent += 1
        if edge['summary_worthy']:
            sum_worthy += 1
        new_edges.setdefault(_id, edge)

    return new_edges, insalientsent, sum_worthy

def get_summary_worth_triple(all_arguments, stopwords, abstract):
    new_all_arguments = []
    for sent_id, arguments in all_arguments:
        new_arguments = []
        for argument in arguments:
            s = [word.lower() for word in (argument[0][0]).split(' ') if word.lower() not in stopwords]
            o = [word.lower() for word in (argument[1][0]).split(' ') if word.lower() not in stopwords]
            p = [word.lower() for word in (argument[2][0]).split(' ') if word.lower() not in stopwords]
            op_words = o + p
            for word in op_words:
                if word in abstract:
                    new_arguments.append(argument)
                    break
        if len(new_arguments) != 0:
            new_all_arguments.append([sent_id, new_arguments])
    if len(new_all_arguments) == 0:
        new_all_arguments = select_largest_group(all_arguments)


    return new_all_arguments



@curry
def process_one(split, stopwords, stemmer, data, ground_truth=False, test=False):
    key = data[0]
    # if int(data[0].split('/')[-1].split('.')[0]) % 1000 == 1:
    #     gc.collect()
    value = data[1]

    _id = int(key.split('/')[-1].split('.')[0])
    print('start processing:', _id)
    # key = OPENIE_DIR + 'temp/' + str(_id) + '.txt'
    # if not data.__contains__(key):
    #     filename = os.path.join(DATA_DIR, split, str(_id) + '.json')
    #     abstract, entities, all_data, article = prepare_data(filename)
    #     all_data['nodes'] = {}
    #     all_data['edges'] = {}
    #     return 0 # return something

    filename = os.path.join(DATA_DIR, split, str(_id) + '.json')
    abstract, entities, all_data, article, extracts = prepare_data(filename, split)
    all_arguments = []
    calibrate = 0
    for sent_id, tuples in value.items():
        sent_id = int(sent_id) - calibrate
        if sent_id > len(article)-1:
            sent_id = len(article)
            flag = 1
            while (flag and sent_id > 1):
                if ' '.join(tuples[0][10].strip('\.| ').split(' ')[:5]) in article[sent_id - 1]:
                    calibrate += 1
                    flag = 0
                sent_id = sent_id - 1
            if sent_id < 1:
                print('bad sample:', _id)
                break
        if not ' '.join(tuples[0][10].strip('\.| ').split(' ')[:5]) in article[sent_id]:
            flag = 1
            while(flag and sent_id > 1):
                if ' '.join(tuples[0][10].strip('\.| ').split(' ')[:5]) in article[sent_id - 1]:
                    calibrate += 1
                    flag = 0
                sent_id = sent_id - 1
            if sent_id < 1:
                print('bad sample:', _id)
                print('article:', article)
                print('tuples:', tuples)
                break

        new_tuples = filter_argument(tuples) # apply rule 0
        final_clusters = find_share_mention(new_tuples)
        arguments = filter_tuples(final_clusters, tuples, entities, sent_id)
        arguments = filter_stopwords(arguments, stopwords)
        all_arguments.append((sent_id, arguments))

    if ground_truth:
        summary = ' '.join(abstract.lower())
        if test:
            before = all_arguments
        all_arguments = get_summary_worth_triple(all_arguments, stopwords, summary)

    pruned_all_arguments = pruning(all_arguments)
    nodes_pruned, edges_pruned = final_process(pruned_all_arguments, entities, article)

    nodes, edges = final_process(all_arguments, entities, article)
    nodes, insalientsent, sum_worthy = make_summary_worth_node(nodes, abstract, stopwords, extracts, stemmer)
    edges, ist, sw = make_summary_worth_edge(edges, abstract, stopwords, extracts, stemmer)

    nodes_pruned, insalientsent_pruned, sum_worthy_pruned = make_summary_worth_node(nodes_pruned, abstract, stopwords, extracts, stemmer)
    edges_pruned, istp, swp = make_summary_worth_edge(edges_pruned, abstract, stopwords, extracts, stemmer)


    all_data['nodes_pruned2'] = nodes_pruned
    all_data['edges_pruned2'] = edges_pruned
    if ground_truth:
        all_data['nodes_sw'] = nodes
        all_data['edges_sw'] = edges

    all_data['nodes'] = nodes
    all_data['edges'] = edges

    if not test:
        with open(os.path.join(WRITE_DIR, split, str(_id) + '.json'), 'w') as f:
            json.dump(all_data, f)
    else:
        print('before:', before)
        print('all_arguments:', all_arguments)
        print(abstract)
    #qgc.collect()
    #print('{} finished'.format(_id))
    # print('all node num:', len(all_data['nodes']))
    # print('node num:', len(nodes))
    return _id, len(nodes), len(nodes_pruned), len(edges), len(edges_pruned), insalientsent, sum_worthy, insalientsent_pruned, sum_worthy_pruned, \
           ist, sw, istp, swp


def process_mp(split, data, stopwords, data_num, stemmer):
    start = time.time()
    with mp.Pool(processes=8) as pool:
        results = list(pool.imap_unordered(process_one(split, stopwords, stemmer, ground_truth=False), list(data.items()), chunksize=1000))
    sents = list(range(data_num))
    processed_art = [result[0] for result in results]
    node_num = [result[1] for result in results]
    node_num_pruned = [result[2] for result in results]
    edge_num = [result[3] for result in results]
    edge_num_pruned = [result[4] for result in results]
    ist = [result[5] for result in results]
    sw = [result[6] for result in results]
    istp = [result[7] for result in results]
    swp = [result[8] for result in results]
    iste = [result[9] for result in results]
    swe = [result[10] for result in results]
    istep = [result[11] for result in results]
    swep = [result[12] for result in results]

    print('average node num:', sum(node_num) / len(node_num))
    print('average edge num:', sum(edge_num) / len(edge_num))
    print('average node num after pruned:', sum(node_num_pruned) / len(node_num_pruned))
    print('average edge num after pruned:', sum(edge_num_pruned) / len(edge_num_pruned))
    print('average sum worthy nodes:', sum(sw) / len(sw))
    print('average in salient sent nodes:', sum(ist) / len(ist))
    print('average sum worthy nodes after pruned:', sum(swp) / len(swp))
    print('average in salient sent nodes after pruned:', sum(istp) / len(istp))
    print('average sum worthy edges:', sum(swe) / len(swe))
    print('average in salient sent edges:', sum(iste) / len(iste))
    print('average sum worthy edges after pruned:', sum(swep) / len(swep))
    print('average in salient sent edges after pruned:', sum(istep) / len(istep))

    extra_data = set(sents) - set(processed_art)
    extra_data = list(extra_data)
    for _id in extra_data:
        #print('None graph id:', _id)
        filename = os.path.join(DATA_DIR, split, str(_id) + '.json')
        abstract, entities, all_data, article, extracts = prepare_data(filename, split)
        all_data['nodes'] = {}
        all_data['edges'] = {}
        all_data['nodes_pruned2'] = {}
        all_data['edges_pruned2'] = {}
        with open(os.path.join(WRITE_DIR, split, str(_id) + '.json'), 'w') as f:
            json.dump(all_data, f)


    print('finished in {}'.format(timedelta(seconds=time.time() - start)))

def process_mp_split(split, data, stopwords, data_num):
    start = time.time()
    data = list(data.items())
    _split = 10
    length = len(data)
    for i in range(_split):
        if i < _split - 1:
            _input = data[:round(length / _split)]
        else:
            _input = data
        with mp.Pool(processes=16) as pool:
            result = list(pool.imap_unordered(process_one(split, stopwords, prune=False, ground_truth=True), _input, chunksize=1000))
        if i < _split - 1:
            data = data[round(length / _split):]
        else:
            del data
        del _input
        gc.collect()
    sents = list(range(data_num))
    print(sum(result) / len(result))

    # extra_data = set(sents) - set(result)
    # extra_data = list(extra_data)
    # for _id in extra_data:
    #     #print('None graph id:', _id)
    #     filename = os.path.join(DATA_DIR, split, str(_id) + '.json')
    #     abstract, entities, all_data, article = prepare_data(filename)
    #     all_data['nodes'] = {}
    #     all_data['edges'] = {}
    #     with open(os.path.join(WRITE_DIR, split, str(_id) + '.json'), 'w') as f:
    #         json.dump(all_data, f)


    print('finished in {}'.format(timedelta(seconds=time.time() - start)))




if __name__ == '__main__':
    stopwords = get_stopwods()
    stemmer = porter.PorterStemmer()
    # for split in ['test', 'val', 'train']:
    for split in ['test']:
        if not os.path.exists(os.path.join(WRITE_DIR, split)):
            os.makedirs(os.path.join(WRITE_DIR, split))
        start = time.time()
        data = read_raw(split)
        print('finish reading data')
        print('time elapsed:', time.time() - start)
        print(len(data))
        # test
        # process_one(split, stopwords, (OPENIE_DIR + 'temp/43.txt', data[OPENIE_DIR + 'temp/43.txt']), prune=False)
        # process_one(split, stopwords, (OPENIE_DIR + 'temp/43.txt', data[OPENIE_DIR + 'temp/43.txt']), prune=False, ground_truth=True, test=True)
        # for i in list(data.items()):
        #     process_one(split, stopwords, i, test=True)

        # train part
        files = glob.glob(os.path.join(DATA_DIR, split, '*'))
        print(len(files))
        gc.collect()
        process_mp(split, data, stopwords, len(files), stemmer)
        #process_mp_split(split, data, stopwords, len(files))



        # post process
        # gc.collect()
        # data = read_raw_2(split)
        # results = []
        # for key, value in list(data.items()):
        #     _id = int(key.split('/')[-1].split('.')[0])
        #     results.append(_id)
        # print(len(results))
        # sents = list(range(len(files)))
        # extra_data = set(sents) - set(results)
        # extra_data = list(extra_data)
        # print('extra data num:', len(extra_data))
        # for _id in extra_data:
        #     filename = os.path.join(DATA_DIR, split, str(_id) + '.json')
        #     abstract, entities, all_data, article = prepare_data(filename)
        #     all_data['nodes_pruned2'] = {}
        #     all_data['edges_pruned2'] = {}
        #     with open(os.path.join(WRITE_DIR, split, str(_id) + '.json'), 'w') as f:
        #         json.dump(all_data, f)




