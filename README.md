# SubRank
The code for the paper "SubRank: Subgraph Embeddings via a Subgraph Proximity Measure", accepted at PAKDD 2020.

Requirements for running the code: python 3, graph-tool 2.27, scikit-learn, C++(11)


Steps for computing embeddings:
1. Compile the file subrank_embeddings.cpp using the makefile
2. Download a directed graph and store it as a list of edges. We share the graphs we used: https://tinyurl.com/subrank
3. For the applications of node clustering/node classification/edge prediction we compute the embeddings of egonetworks.
By default we use egonetworks of size 1. To compute the embedding of ego networks, we first compute the proximity of egonetworks.
Run compute_embeddings with the option --help to get information about the parameters.
4. Finaly, we can examine the quality of the embeddings with the embedding_evaluation. Give the option --help for more info on the parameters. 
