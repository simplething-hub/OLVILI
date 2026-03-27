import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import scipy.io as sio
import scipy.sparse as ss
import random
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np

def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    # start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs + 1, algorithm='ball_tree').fit(features)
    adj_construct = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>
    adj = ss.coo_matrix(adj_construct)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if prunning_one:
        # Pruning strategy 1
        original_adj = adj.A
        judges_matrix = original_adj == original_adj.T
        adj = original_adj * judges_matrix
        adj = ss.csc_matrix(adj)
    # obtain the adjacency matrix without self-connection
    adj = adj - ss.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = ss.coo_matrix(adj)
        adj.eliminate_zeros()

    # print("The construction of Laplacian matrix is finished!")
    # print("The time cost of construction: ", time.time() - start_time)
    adj = ss.coo_matrix(adj)
    return adj

def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict

def generate_partition(labels, ratio, seed):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {} ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1) # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    # print(index)
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
            total_num -= 1
        else:
            p_unlabeled.append(index[idx])
    return p_labeled, p_unlabeled

def construct_adj_hat(adj):
    """
        construct the Laplacian matrix
    :param adj: original adj matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    # adj = ss.coo_matrix(adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    rowsum = np.array(adj_.sum(1)) # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = ss.eye(adj.shape[0]) - adj_wave
    return adj_wave, lp
def load_data(args,device):
    df = pd.read_excel('./original_data_p6.xlsx')
    X = df.iloc[1:, :-1]
    y = df.iloc[1:, -1].values
    # 归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    view = 3
    num = X.shape[0]  
    fea_num = X.shape[1]

    adj = construct_adjacency_matrix(X, args.k, False, True, 2)
    # adj = ss.coo_matrix(adj)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj_hat, lp = construct_adj_hat(adj)
    idx_labeled, idx_unlabeled = generate_partition(labels=y, ratio=0.8, seed=42)
    # 划分视图特征
    # X_1 = X[:, 0:15]
    # X_2 = X[:, 15:17]
    # X_3 = X[:, 17:]
    # X = [X_1, X_2, X_3]
    y = y.astype(np.int64)
    return X, y, idx_labeled, idx_unlabeled, view, fea_num, num, adj



