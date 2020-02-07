# here we compute VERSE embeddings and SubRank transductive embeddings

import os
import sys
import argparse

def generate_verse_embeddings(path_to_graph, output_embeddings):
    filenamebcsr = path_to_graph+".bcsr"
    conversion_str = "python convert.py " + path_to_graph +" "+ filenamebcsr
    os.system(conversion_str)
    verse_str = "./verse -input " + filenamebcsr + " -output "+ output_embeddings + " -dim 128 -threads 10 -nsamples 3"
    os.system(verse_str)

def generate_subrank_embeddings(path_to_graph, filename_proximity, output_embeddings):
    filenamebcsr = path_to_graph + ".bcsr"
    conversion_str = "python convert.py " + path_to_graph +" "+ filenamebcsr
    os.system(conversion_str)
    verse_str = "./verse_distributionsample -input " + filenamebcsr +" -pprfile " + filename_proximity + " -output "+ output_embeddings +" -dim 128 -threads 1 -nsamples 3"
    os.system(verse_str)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog='compute_embeddings',
                                        usage='compute_embeddings path_to_graph (path_to_subgraph_proximity) name_output_file',
                                        description='It uses VERSE as a subroutine to compute subgraph embedding. It can also use VERSE to compute node embeddings.')
    my_parser.add_argument("-i", "--input", required=True,
                    help="path to input graph")
    my_parser.add_argument("-o", "--output", required=True,
                    help="path to output embeddings")
    my_parser.add_argument("-s", "--subroutine", required=True,
                    help="subroutine to run: verse or subrank")
    my_parser.add_argument("-p", "--proximity", required=False,
                           help="path for proximity file for subrank")

    args = vars(my_parser.parse_args())
    print(args) 
    if args.get('subroutine') == "subrank":
        generate_subrank_embeddings(args.get('input'), args.get('proximity'), args.get('output'))
    elif args.get('subroutine') == "verse":
        generate_verse_embeddings(args.get('input'), args.get('output'))
    else:
        print("the subroutine name is not correct")
