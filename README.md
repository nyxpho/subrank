# subrank
The code for the paper "SubRank: Subgraph Embeddings via a Subgraph Proximity Measure", accepted at PAKDD 2020.

Requirements for running the code: python 3, graph-tool, scikit-learn, C++(11)

Steps for computing embeddings:
1. Compile the file verse_distributionsample.cpp using the makefile
2. Download a directed graph and store it as a list of edges.
3. For the applications of node clustering/node classification/edge prediction we compute the embeddings of egonetworks.
By default we use egonetworks of size 1. To compute the embedding of ego networks, we first compute the proximity of egonetworks.

