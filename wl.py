import argparse
import numpy as np
from util import load_data, separate_data
from grakel import GraphKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def convert(graphs):
    X = []
    y = []
    for graph in graphs:
        edge = {i: neighbors for i, neighbors in enumerate(graph.neighbors)}
        node_label = {i: label for i, label in enumerate(graph.node_tags)}
        X.append([edge, node_label])
        y.append(graph.label)
    return X, y


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='WL subtree kernel')
    parser.add_argument('--dataset',
                        type=str,
                        default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument(
        '--fold_idx',
        type=int,
        default=0,
        help='the index of fold in 10-fold validation. Should be less then 10.'
    )
    parser.add_argument('--iter',
                        type=int,
                        default=5,
                        help='Number of iteration for the WL')
    parser.add_argument('--normalize',
                        action="store_true",
                        help='normalize the feature or not')
    parser.add_argument('--filename', type=str, default="", help='output file')
    args = parser.parse_args()

    np.random.seed(0)
    graphs, num_classes = load_data(args.dataset, False)

    ##10-fold cross validation, consider the particular fold.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    #SVM hyper-parameter to tune
    C_list = [0.01, 0.1, 1, 10, 100]
    X_train, y_train = convert(train_graphs)
    X_test, y_test = convert(test_graphs)

    wl_kernel = GraphKernel(kernel=[{
        "name": "weisfeiler_lehman",
        "niter": args.iter
    }, {
        "name": "subtree_wl"
    }],
                            normalize=args.normalize)
    K_train = wl_kernel.fit_transform(X_train)
    K_test = wl_kernel.transform(X_test)

    train_acc = []
    test_acc = []
    for C in C_list:
        clf = SVC(kernel='precomputed', C=C)
        clf.fit(K_train, y_train)
        y_pred_test = clf.predict(K_test)
        y_pred_train = clf.predict(K_train)
        train_acc.append(accuracy_score(y_train, y_pred_train) * 100)
        test_acc.append(accuracy_score(y_test, y_pred_test) * 100)

    print(train_acc)
    print(test_acc)

    if not args.filename == "":
        np.savetxt(args.filename, np.array([train_acc, test_acc]).transpose())


if __name__ == '__main__':
    main()