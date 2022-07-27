"""
Implementation of group-based and kernel-based weighted IAutoRec. The default method is kW-IAutoRec, swift to gW-IAutoRec by setting gw = True.

"""

import tensorflow as tf
import time
import numpy as np
import scipy
import sys

from utils.evaluation.RatingMetrics import *


class IAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.005, reg_rate=20, epoch=5000, batch_size=500,
                 verbose=False, gW=False, display_step=1000):  
        
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.gW = gW
        self.display_step = display_step

        print("IAutoRec.")

    def build_network(self, hidden_neuron=500):  

        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):
        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                               self.keep_rate_net: 0.95
                                               })
            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    
    
    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            #error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = RMSE(error, len(test_set))
            #mae = MAE(error_mae, len(test_set))
        print("RMSE:" + str(rmse)) #+ "; MAE:" + str(mae))
        return rmse
    
    
    def execute(self, train_data, vali_data, test_data, test_data_new):
        train_data = self._data_process(train_data)
        
        # eG-AutoRec: Filled new users vanilla group mean
        if self.gW:
            group_u = np.repeat(np.arange(10), 100)
            new_idx = np.where(np.sum(train_data, axis=1)==0)
            for idx in new_idx[0]:
                g_u = group_u[idx]
                item = train_data[group_u==g_u,]
                mu = np.sum(item,axis=0)/np.sum(abs(scipy.sign(item)), axis=0)
                # randomly add 10% imputed values to training
                sparsity = np.random.choice(np.arange(2), size=len(train_data[0,]), p=[0.9, 0.1])
                train_data[idx,] = np.nan_to_num(np.multiply(mu, sparsity)) 



        # wG-AutoRec: kernel weighted
        else:
            group_u = np.repeat(np.arange(10), 100)
            pos = np.arange(100)/100
            new_idx = np.where(np.sum(train_data, axis=1)==0)
            sigma = 0.1
            for idx in new_idx[0]:
                g_u = group_u[idx]
                item = train_data[group_u==g_u,]
                item_mask = abs(scipy.sign(item))
                pos0 = pos[idx%100]
                dist = (pos - pos0)**2
                weights = np.exp(-dist/2/sigma**2).reshape(100,-1)
                item_w = item*weights
                item_mask_w = item_mask*weights
                mu = np.sum(item_w,axis=0)/np.sum(item_mask_w, axis=0)
                # randomly add 10% imputed values to training
                sparsity = np.random.choice(np.arange(2), size=len(train_data[0,]), p=[0.9, 0.1])
                train_data[idx,] = np.nan_to_num(np.multiply(mu, sparsity)) 

        
        self.train_data = np.nan_to_num(train_data)
        self.train_data_mask = scipy.sign(self.train_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        best = [9999, -1, 9999, 9999]
        for epoch in range(self.epochs):              

            self.train(train_data)
            print("Epoch: %04d; " % (epoch), end='')
            cur = self.test(vali_data)  
            if cur < best[0]: 
                best = [cur, epoch, self.test(test_data), self.test(test_data_new)]
            elif epoch - best[1] > 10:
                break
        
        print ("End of training:")
        print ("Epoch: %d" % best[1])
        print ("RMSE of old users: %0.5f" %best[2])
        print ("RMSE of new users: %0.5f" %best[3])
                

    def save(self, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i))
        return output


    