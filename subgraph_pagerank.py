import numpy as np
from graph_operations import *
from joblib import Parallel, delayed
import multiprocessing
import argparse
import random

import numpy as np
import graph_tool as gt
import graph_tool.centrality as gc
from graph_tool import GraphView
from scipy.sparse import csr_matrix


'''
This function reads a graph from the file and renumbers the vertices.
The original IDs can be retrieved using the vertices properties
This code makes the assumption that the ids of the nodes start from 0 and each node has at least one outgoing or ingoing link.
'''


def read_graph(filename, direction):
    graph = gt.load_graph_from_csv(filename, directed=direction, hashed=True, string_vals=True,
                                   csv_options={'delimiter': ' '})
    return graph


'''
For each node in the graph we return its ego network of size 1.
An ego network is represented as a graph tool object.
Very important, we need to take care of using the correct indices as graph tool
might use new vertex indices.
'''


def egonets(graph, direction):
    egonets = dict()
    for node in graph.vertices():
        mask = graph.new_vertex_property("bool")
        for u in node.out_neighbours():
            mask[u] = True
        mask[node] = True
        label_node = graph.vertex_properties["name"][node]
        egonets[label_node] = GraphView(graph, vfilt=mask)
    return egonets

def read_cascades(filename, graph, direction):
    rb = open(filename, 'r')
    cascades = dict()
    for line in rb.readlines():
        parse = line.strip().split('\t')
        cascades[parse[0]] = []
        for edges in parse[4].split(' '):
            edges = edges.split(':')
            cascades[parse[0]].append((edges[0], edges[1], float(edges[2])))

    for ids in cascades:
        list_edges = cascades[ids]
        cascades[ids] = gt.Graph(directed=direction)
        cascades[ids].vertex_properties["name"] = cascades[ids].add_edge_list(list_edges, hashed = True, string_vals=True)

    return cascades

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
    pr_list = []
    norm_dict = normalize_dictionary(vec_dict)
    for poz in norm_dict:
        poz_initial = subgraph.vertex_properties["name"][poz]
        pr_list.append((poz_initial, norm_dict[poz]))
    pr_list = sorted(pr_list, key=lambda tup: tup[1], reverse=True)
    return pr_list


def print_subgraph_pr(graph, subgraphs, outfile):
    wb = open(outfile, 'w')
    for i in subgraphs.keys():
        PR_s = PR_subgraph(graph, subgraphs[i], epsilon, threshold)
        wb.write(str(i))
        for t in PR_s:
            if t[1] > 0:
                wb.write(" " + str(t[0]) + " " +  str(t[1]))
        wb.write("\n")

    wb.close()



if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog='subgraph_proximity',
                                        usage='subrank_proximity path_to_graph path_output_proximity_file',
                                        description='This program computes the proximity between ego networks and store it in a file. ')
    my_parser.add_argument("-i", "--input", required=True,
                           help="path to input graph")
    my_parser.add_argument("-o", "--output", required=True,
                           help="path for output pagerank of subgraph file necessary for subrank")
    args = vars(my_parser.parse_args())
    g = read_graph(args.get("input"), True)
    random.seed(42)
    subgraphs = egonets(g, True)
    #this code can be easily changed to read cascades from a file (use function read cascades)
    epsilon = 1.0 / g.num_vertices()
    threshold = epsilon
    print_subgraph_pr(g, subgraphs, args.get("output"))
