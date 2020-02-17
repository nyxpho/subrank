import numpy as np
from graph_operations import *
from joblib import Parallel, delayed
import multiprocessing
import argparse


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
    PPR = PPR_graph_tocsr(graph, epsilon, threshold)
    for i in subgraphs.keys():
        PR_s = PR_subgraph_tocsr(graph, subgraphs[i], epsilon, threshold)
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


'''
We give a second implementation of the subrank proximity, using random walks.
This implementation is faster and should be used for large networks.
'''
def random_walk_with_restart(graph, node_start, alpha=0.85)
    prob = np.random()
    node_end = node_start
    while prob < alpha:
        neigh = graph.get_out_neighbors(node_end)
        poz = np.randint(0, neigh.size) # the random walk is performed in an unweighted network
        node_end = neigh[poz]
        prob = np.random()
    return node_end

def generate_pairs_proximity(graph, subgraphs, num_pairs, epsilon, threshold, out_file):
    wb = open(out_file, "r")
    all_PRs = dict()
    for i in subgraphs.keys():
        PR_s = PR_subgraph_tocsr(graph, subgraphs[i], epsilon, threshold)
        all_PRs[i] = PR_s

    name_to_index = dict()
    for i in range(graph.num_vertices):
        name_to_index[graph.vertex_properties["name"][i]] = i

    for i in range(num_pairs):
        # select a starting ego network
        index = np.randint(graph.num_vertices())
        ego_start = graph.vertex_properties["name"][index]

        # select a node node_start according to its PR in the selected ego network
        pr_s1_list = all_PRs[ego_start][1]
        pr_node = np.random()
        sum_pr = 0
        node_start = ego_start
        for t in pr_s1_list:
            if sum_pr <= pr_node < sum_pr + t[1]:
                node_start = t[0]
                break
            sum_pr = sum_pr + t[1]

        # perform from the node_start a random walk in the graph until node_end
        node_end = random_walk_with_restart(graph, name_to_index[node_start])

        # retrieve all the ego networks containing node_end and its probability in these ego networks
        in_neigh = graph.get_in_neighbors(node_end)
        node_end_name = graph.vertex_properties["name"][node_end]
        prob_s2 = dict()
        for index in in_neigh:
            pr_s2_dict = all_PRs[graph.vertex_properties["name"][index]][0]
            prob_s2[index] = pr_s2_dict[node_end_name]
        prob_s2_norm = normalize_dictionary(prob_s2)
        list_prob_s2 = [(k, v) for k, v in prob_s2_norm.items()]
        
        # select the finish ego network according to the probability of node_end belonging to it
        prob_s2_list = sorted(list_prob_s2 , key = lambda tup: tup[1], reverse=True)
        pr_node = np.random()
        sum_pr = 0
        ego_end = node_end_name
        for t in prob_s2_list:
            if sum_pr <= pr_node < sum_pr + t[1]:
                ego_end = graph.vertex_properties["name"][t[0]]
                break
            sum_pr = sum_pr + t[1]

        wb.write(str(ego_start) + " " + str(ego_end) + "\n")

    wb.close()

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
    #save_rank_proximities(g, subgraphs, epsilon, threshold, args.get("output"))
    generate_pairs_proximity(g, subgraphs, 10000, epsilon, threshold, args.get("output")):