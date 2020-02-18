import numpy as np
import graph_tool as gt
import graph_tool.centrality as gc
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
    for item in graph.edges():
        vertex_1 = graph.vertex_properties["name"][item.source()]
        vertex_2 = graph.vertex_properties["name"][item.target()]
        if vertex_1 in egonets.keys():
            egonets[vertex_1].append((vertex_1, vertex_2))
        else:
            egonets[vertex_1] = []
            egonets[vertex_1].append((vertex_1, vertex_2))

    for i in graph.vertices():
        vertex = graph.vertex_properties["name"][i]  # we add the real ids of the vertices

        if vertex not in egonets.keys():
            egonets[vertex] = gt.Graph(directed=direction)
            egonets[vertex].vertex_properties["name"] = egonets[vertex].add_edge_list([(vertex, vertex)], hashed=True,
                                                                                      string_vals=True)
        else:
            list_ego = egonets[vertex]
            egonets[vertex] = gt.Graph(directed=direction)
            egonets[vertex].vertex_properties["name"] = egonets[vertex].add_edge_list(list_ego, hashed=True,
                                                                                      string_vals=True)

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


def PR_subgraph_tocsr(graph, subgraph, eps, threshold):
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

def PR_subgraph_todictandlist(graph, subgraph, eps, threshold):
    pr = gc.pagerank(subgraph, epsilon=eps)
    vec = pr.a
    vec_dict = dict()
    index = 0
    for value in vec:
        vec_dict[index] = value
        index += 1
    #pruned = prune_elements(vec_dict, threshold)  # we remove PR values smaller than a threshold for space efficiency
    pr_dict = dict()
    pr_list = []
    norm_dict = normalize_dictionary(vec_dict)
    for poz in norm_dict:
        poz_initial = subgraph.vertex_properties["name"][poz]
        pr_dict[poz_initial] = norm_dict[poz]
        pr_list.append((poz_initial, norm_dict[poz]))
    pr_list = sorted(pr_list, key=lambda tup: tup[1], reverse=True)
    return (pr_dict, pr_list)
'''
We compute the personalised pagerank values between two nodes in the graph.
'''


def PPR_graph_tocsr(graph, eps, threshold):
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
