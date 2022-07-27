import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_data_rating(path="../data/full.data890.csv", header=0, seed = 1, n_new=50,
                     vali_size=0.15, test_size=0.1, sep=","):
    '''
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param vali_size: the validation ratio, default 0.15
    :param test_size: the test ratio, default 0.1
    :param sep: the seperator for csv colunms, defalut space
    :return:
    '''

    df = pd.read_csv(path, sep=sep, header=0, engine='python')
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    
    seed = int(path[-7:-4]) + seed  
    random.seed(seed)
    new_idx = random.sample(range(1,(n_users+1)), n_new)

    train_vali_data, test_data = train_test_split(df.loc[~df['user_id'].isin(new_idx),], test_size=test_size, random_state=3)    
    train_data, vali_data = train_test_split(train_vali_data, test_size=vali_size, random_state=3)
    
    train_data = pd.DataFrame(train_data)
    vali_data = pd.DataFrame(vali_data)
    test_data = pd.DataFrame(test_data)
    test_data_new = df.loc[df['user_id'].isin(new_idx),]
    
    print ("The number of training: ", len(train_data))
    print ("The number of old users: ", len(np.unique(test_data['user_id'])))
    print ("The number of new users: ", n_users-len(np.unique(train_data['user_id'])))
    print ("The number of old users rating: ", len(test_data))
    print ("The number of new users rating: ", len(test_data_new))
    

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    
    vali_row = []
    vali_col = []
    vali_rating = []
    for line in vali_data.itertuples():
        vali_row.append(line[1] - 1)
        vali_col.append(line[2] - 1)
        vali_rating.append(line[3])
    vali_matrix = csr_matrix((vali_rating, (vali_row, vali_col)), shape=(n_users, n_items))
    
    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_row_new = []
    test_col_new = []
    test_rating_new = []
    for line in test_data_new.itertuples():
        test_row_new.append(line[1] - 1)
        test_col_new.append(line[2] - 1)
        test_rating_new.append(line[3])
    test_matrix_new = csr_matrix((test_rating_new, (test_row_new, test_col_new)), shape=(n_users, n_items))
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), vali_matrix.todok(), test_matrix.todok(), test_matrix_new.todok(), n_users, n_items