
import numpy as np

import argparse
from graph_operations import *

def compute_avg(graph, subgraphs, fileout, embedding_file, embedding_dim):
    embeddings = np.fromfile(embedding_file, np.float32).reshape(-1, embedding_dim)
    wb = open(fileout, "w")
    for i in subgraphs.keys():
        subgr = subgraphs[i]
        emb = np.zeros(embedding_dim)
        for v in subgr.vertices():
            emb += graph.vertex_properties["name"][v]
        emb /= len(subgr.vertices())
        wb.print(graph.vertex_properties["name"][v] + " ".join(emb) + "\n")
    wb.close()

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog='compute_average_verse',
                                        usage='compute_average_verse path_to_graph  name_output_file',
                                        description='It computes VERSE average ego network embeddings.')
    my_parser.add_argument("-i", "--input", required=True,
                    help="path to input graph")
    my_parser.add_argument("-o", "--output", required=True,
                    help="path to output embeddings")
    my_parser.add_argument("-e", "--embeddings", required=True,
                    help="subroutine to run: verse or subrank")

    args = vars(my_parser.parse_args())
    g = read_graph(args.get("input"), True)
    subgraphs = egonets(g, True)
    print(args)
    compute_avg(graph, subgraphs, args.get("output"), args.get("embeddings"), 128)