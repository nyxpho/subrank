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
										usage='%(prog) path_to_graph (path_to_subgraph_proximity) name_output_file',
										description='It uses VERSE as a subrutine to compute subgraph embedding. It can also use VERSE to compute node embeddings.')
	my_parser.add_argument("-i", "--input", required=True,
					help="path to input graph")
	my_parser.add_argument("-o", "--output", required=True,
					help="path to output embeddings")
	my_parser.add_argument("-s", "--subrutine", required=True,
					help="subrutine to run: verse or subrank")
	my_parser.add_argument("-p", "--proximity", required=False,
						   help="path for proximity file for subrank")

	args = vars(ap.parse_args())
	if args.get(s) == "subrank":
		generate_subrank_embeddings(args.get(i), sys.argv[2], sys.argv[3])
	elif args.get(s) == "verse":
		generate_verse_embeddings(sys.argv[1], sys.argv[2])
    else:
		print("the subrutine name is not correct")