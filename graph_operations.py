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

'''
This function retrieves the test and training edge sets for link prediction.
VERY IMPORTANT: embeddings should be learned on a graph containing only the train edges.
'''


def linkprediction_testtrain(graph, percentage):
    count = graph.num_edges() * percentage
    test_edgelist = []
    train_edgelist = []
    degrees = graph.get_total_degrees(graph.get_vertices())
    for item in graph.edges():
        tuple_edge = (graph.vertex_properties["name"][item.source()], graph.vertex_properties["name"][item.target()])
        if count > 0:
            if degrees[int(item.target())] > 1 and degrees[int(item.source())] > 1 and int(item.source()) != int(item.target()):
                count -= 1
                degrees[int(item.source())] -= 1
                degrees[int(item.target())] -= 1
                test_edgelist.append(tuple_edge)
            else:
                train_edgelist.append(tuple_edge)
        else:
            train_edgelist.append(tuple_edge)

    return test_edgelist, train_edgelist



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




