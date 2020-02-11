import sys
import numpy as np
import graph_tool as gt
import graph_tool.centrality as gc
from scipy.sparse import csr_matrix
from graph_operations import read_graph, egonets
from joblib import Parallel, delayed
import multiprocessing
import argparse

def normalize_dictionary(v):
    norm_of_v = 0.0
    for k in v:
        norm_of_v = norm_of_v + v[k]
    v_norm = dict()
    if norm_of_v == 0:
        for k in v:
            v_norm[k] = 1.0 / len(v)
    else:
        for k in v:
            v_norm[k] = v[k] / norm_of_v
    return v_norm


def prune_elements(vec, threshold):
    new_dict = dict()
    for (k, v) in vec.items():
        item = v
        if item > threshold:
            new_dict[k] = v
    return normalize_dictionary(new_dict)


'''
We compute the pagerank of nodes in a induced subgraph.
'''


def PR_subgraph(graph, subgraph, eps, threshold):
    pr = gc.pagerank(subgraph, epsilon=eps)
    vec = pr.a
    vec_dict = dict()
    index = 0
    for value in vec:
        vec_dict[index] = value
        index += 1
    pruned = prune_elements(vec_dict, threshold)  # we remove PR values smaller than a threshold for space efficiency
    data = []
    row_ind = []
    col_ind = []
    for poz in pruned:
        poz_initial = int(subgraph.vertex_properties["name"][poz])
        row_ind.append(0)
        col_ind.append(poz_initial)
        data.append(pruned[poz])

    return csr_matrix((data, (row_ind, col_ind)), shape=(1, graph.num_vertices()))


'''
We compute the personalised pagerank values between two nodes in the graph.
'''


def PPR_graph(graph, eps, threshold):
    personalise = np.zeros(graph.num_vertices())
    prop = graph.new_vertex_property("int16_t")
    PPRs = []
    row_ind = []
    col_ind = []

    for i in range(graph.num_vertices()):
        personalise[i] = 1
        prop.a = personalise
        pr = gc.pagerank(graph, pers=prop, epsilon=eps)
        vec = pr.a
        vec_dict = dict()
        index = 0
        for value in vec:
            vec_dict[index] = value
            index += 1
        pruned = prune_elements(vec_dict,
                                threshold)  # we remove PPR values smaller than a threshold for space efficiency
        for poz in pruned:
            i_initial = int(graph.vertex_properties["name"][i])
            j_initial = int(graph.vertex_properties["name"][poz])
            row_ind.append(i_initial)
            col_ind.append(j_initial)
            PPRs.append(pruned[poz])
        personalise[i] = 0

    return csr_matrix((PPRs, (row_ind, col_ind)), shape=(graph.num_vertices(), graph.num_vertices()))


'''
This function will return a vector of proximities from
source to destination. It implements the proximity measure
we propose. More precisely, given S1 and S2, we have:
P(S1,S2) = sum_{i in S1}sum_{j in S2} PR(i, S1) * PPR(i,j,G)*PR(j,S2)

To write this as matrix multiplication we have the following matrices,
given that S1 has n1 nodes and S2 has n2 nodes.
M1 (1,n1) = each column i has the value PR(i)
M2 (n1, n2) = all the PPR(i,j) values
M3 (n2, 1) = each line has the value PR (j)
M2*M3 (n1,1) = a cell i contains sum_{j in S2} PPR(i,j)PR(j)

To simplify, we use the dimension n, as in the sparse format 0's are not stored.

We give in input the sparse matrix PPRs and the dictionary of sparse matrices PRs
and the matrices index source and destinations
'''


def product(M1, M2, j):
    product = M1 * M2.transpose()
    p_dict = product.todok()
    value = 0
    if len(p_dict.items()) > 0:
        value = list(p_dict.items())[0][1]
    return (j, value)


def save_rank_proximities(graph, subgraphs, epsilon, threshold, filename):
    all_PRs = dict()
    PPR = PPR_graph(graph, epsilon, threshold)
    for i in subgraphs.keys():
        PR_s = PR_subgraph(graph, subgraphs[i], epsilon, threshold)
        all_PRs[i] = PR_s
    print("computed all PR and PPR values")
    sub_rank = []
    row_ind = []
    col_ind = []
    num_cores = multiprocessing.cpu_count()
    for s1 in subgraphs.keys():
        prox = dict()
        M1 = all_PRs[s1] * PPR
        results = Parallel(n_jobs=num_cores)(delayed(product)(M1, all_PRs[s2], s2) for s2 in subgraphs.keys())
        # print results
        for (k, v) in results:
            prox[k] = v
        prox = normalize_dictionary(prox)
        pruned = prune_elements(prox, threshold)
        for poz in pruned:
            row_ind.append(s1)
            col_ind.append(poz)
            sub_rank.append(pruned[poz])

    with open(filename, 'w') as f:
        for i in row_ind:
            f.write('%s ' % i)
        f.write('\n')
        for i in col_ind:
            f.write('%s ' % i)
        f.write('\n')
        for i in sub_rank:
            f.write('%0.8f ' % i)
        f.write('\n')


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog='subgraph_proximity',
                                        usage='subrank_proximity path_to_graph path_output_proximity_file',
                                        description='This program computes the proximity between ego networks and store it in a file. ')
    my_parser.add_argument("-i", "--input", required=True,
                           help="path to input graph")
    my_parser.add_argument("-o", "--output", required=True,
                           help="path for output proximity file necessary for subrank")
    args = vars(my_parser.parse_args())
    g = read_graph(args.get("input"), True)
    subgraphs = egonets(g, True)
    epsilon = 1.0 / g.num_vertices()
    threshold = epsilon
    save_rank_proximities(g, subgraphs, epsilon, threshold, args.get("output"))
