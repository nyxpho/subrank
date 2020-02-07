import csv

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def read_embeddings(filein):
    rb = open(filein, 'r')
    emb = dict()
    for line in rb.readlines():
        elem = line.strip().split(' ')
        e = np.fromstring(' '.join(elem[1:]), dtype=np.float, sep=' ')
        emb[int(elem[0])] = e
    rb.close()
    return emb


def clustering(label_file, embedding_file, embedding_dim, clusters):
    print('performing kmeans clustering -------------------------------------------')

    embeddings = np.fromfile(embedding_file, np.float32).reshape(-1, embedding_dim)
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(embeddings)
    node_labels = kmeans.labels_

    with open(label_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        label_list = list(reader)

    labels = []
    for item in label_list:
        labels.append(int(item[1]))

    nmi_score = normalized_mutual_info_score(node_labels, labels)
    adj_score = adjusted_mutual_info_score(node_labels, labels)

    print(nmi_score)
    print(adj_score)

    return nmi_score


def node_classification(label_file, embedding_file, embedding_dim):
    print("running node_classification ---------------------------------------")

    embeddings = np.fromfile(embedding_file, np.float32).reshape(-1, embedding_dim)
    scaler = StandardScaler()
    scaler.fit(embeddings)
    x_data = scaler.transform(embeddings)
    with open(label_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        label_list = list(reader)

    y_labels = []
    for item in label_list:
        y_labels.append(int(item[1]))

    y_labels = np.array(y_labels)
    parameter_space = {'estimator__kernel': ['rbf', 'linear', 'sigmoid', 'poly'], 'estimator__gamma': [1e-3, 1e-4],
                       'estimator__C': [1, 10, 100, 1000], 'estimator__probability': [True],
                       'estimator__max_iter': [1000, 5000]}

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.2, random_state=42,
                                                        stratify=y_labels)

    model = OneVsOneClassifier(SVC())
    clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=5, scoring='f1_micro')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    macro_score = f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred))
    micro_score = f1_score(y_test, y_pred, average='micro', labels=np.unique(y_pred))
    print('F1-micro / F1-macro scores')
    print(micro_score)
    print(macro_score)

    return micro_score, macro_score


def cascade_prediction(train_file, test_file, val_file, embedding_file, embedding_dim):
    embeddings = np.fromfile(embedding_file, np.float32).reshape(-1, embedding_dim)
    label = 0

    x_test = []
    y_test = []
    index = 0
    rb = open(test_file, 'r')
    for line in rb.readlines():
        elems = line.strip().split('\t')
        x_test.append(embeddings[index])
        index += 1
        value = int(elems[-1].split(' ')[label])
        value = np.log(value + 1.0) / np.log(2.0)
        y_test.append(value)
    rb.close()

    x_train = []
    y_train = []
    rb = open(train_file, 'r')
    for line in rb.readlines():
        elems = line.strip().split('\t')
        x_train.append(embeddings[index])
        index += 1
        value = int(elems[-1].split(' ')[label])
        value = np.log(value + 1) / np.log(2.0)
        y_train.append(value)
        rb.close()

    x_val = []
    y_val = []
    rb = open(val_file, 'r')
    for line in rb.readlines():
        elems = line.strip().split('\t')
        x_val.append(embeddings[index])
        index += 1
        value = int(elems[-1].split(' ')[label])
        value = np.log(value + 1) / np.log(2.0)
        y_val.append(value)
    rb.close()

    parameter_space = {
        'hidden_layer_sizes': [(64, 32, 16), (64, 32, 64), (64,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.05, 0.01, 0.5, 1],
        'learning_rate': ['constant', 'adaptive'],
        'shuffle': [False], 'random_state': [0],
        'max_iter': [200]}
    x = x_train + x_val
    y = y_train + y_val
    fold = [-1 for i in x_train]
    fold.extend([0 for i in x_val])
    model = MLPRegressor()
    ps = PredefinedSplit(test_fold=fold)
    clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=ps)
    clf.fit(x, y)
    print(clf.get_params())
    y_pred = clf.predict(x_test)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(prog='embedding_evaluation',
                                        usage='%(prog) application embedding_file (needed for clustering: number_clusters, needed for node classification/edge prediction: label_file, needed cascade prediction: train/test/val label files) ',
                                        description='This program takes embeddings and evaluates them on different downstream applications: clustering, node clasification, link prediction and for cascade prediction. ')
    my_parser.add_argument("-e", "--embeddings", required=True,
                           help="path to embedding file")
    my_parser.add_argument("-a", "--application", required=True,
                           help="Application: kmeans/node_class/link_pred/cascade_pred")
    my_parser.add_argument("-k", "--clusters", required=False,
                           help="number of clusters")
    my_parser.add_argument("-l", "--label", required=False,
                           help="label file for clustering")
    my_parser.add_argument("-tr", "--train_label", required=False,
                           help="train label file for cascade prediction")
    my_parser.add_argument("-te", "--test_label", required=False,
                           help="test label file for cascade prediction")
    my_parser.add_argument("-v", "--val_label", required=False,
                           help="val label file for cascade prediction")
    args = vars(my_parser.parse_args())
    if args.get(a) == "kmeans":
        clustering(args.get("l"), args.get("e"), 128, int(args.get("k")))
    elif args.get(a) == "node_class":
        node_classification(args.get("l"), args.get("e"), 128)
    elif args.get(a) == "cascade_pred":
        cascade_prediction(args.get("tr"), args.get("te"), args.get("v"), args.get("e"), 128)
    else:
        print("no application found")