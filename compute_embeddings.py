# here we compute SubRank embeddings

import os
import sys
import argparse

def generate_subrank_embeddings(path_to_graph, output_embeddings):
    filenamebcsr = path_to_graph + ".bcsr"
    conversion_str = "python convert.py " + path_to_graph +" "+ filenamebcsr
    os.system(conversion_str)
    pagerank_str = "python subgraph_pagerank.py --input " + path_to_graph + " --output " + path_to_graph + ".pr"
    print(pagerank_str)
    os.system(pagerank_str)
    embedding_str = "./subrank_embeddings -graph " + filenamebcsr +" -prfile " + path_to_graph + ".pr" + " -embedding "+ output_embeddings +" -dim 128 -threads 10 -nsamples 3"
    print(embedding_str)
    os.system(embedding_str)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog='compute_embeddings',
                                        usage='compute_embeddings path_to_graph name_output_embeddings',
                                        description='This program computes subgraph embedding. By default, we suppose that the subgraphs are the ego network of size 1.')
    my_parser.add_argument("-i", "--input", required=True,
                    help="path to input graph")
    my_parser.add_argument("-o", "--output", required=True,
                    help="path to output embeddings")

    args = vars(my_parser.parse_args())
    generate_subrank_embeddings(args.get("input"), args.get("output"))
