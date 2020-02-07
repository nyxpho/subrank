import sys
import graph_tool as gt

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
    for item in graph.iter_edges():
        vertex_1 = graph.vertex_properties["name"][item[0]]
        vertex_2 = graph.vertex_properties["name"][item[1]]
        if vertex_1 in egonets.keys():
            egonets[vertex_1].append((vertex_1, vertex_2))
        else:
            egonets[vertex_1] = []
            egonets[vertex_1].append((vertex_1, vertex_2))

    for i in graph.iter_vertices():
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
    for item in graph.iter_edges():
        tuple_edge = (graph.vertex_properties["name"][item[0]], graph.vertex_properties["name"][item[1]])
        if count > 0:
            if degrees[int(item[1])] > 1 and degrees[int(item[0])] > 1 and int(item[0]) != int(item[1]):
                count -= 1
                degrees[int(item[0])] -= 1
                degrees[int(item[1])] -= 1
                test_edgelist.append(tuple_edge)
            else:
                train_edgelist.append(tuple_edge)
        else:
            train_edgelist.append(tuple_edge)

    return test_edgelist, train_edgelist
