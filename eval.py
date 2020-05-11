import os
from os.path import isfile, join
mypath = "../subgraphEmbeddings/baselines/node2vec/dblpd"
onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]

for p in [0.01]:
    for f in onlyfiles:
        os.system("python3 embedding_evaluation.py -e " +  mypath + "/" + f + " -a node_class -l ../subgraphEmbeddings/data/dblp_labels -pr " + str(p) + " >> output.txt")
