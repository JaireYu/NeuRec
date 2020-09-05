#!/usr/local/bin/python
from model.AbstractRecommender import AbstractRecommender
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from util.Logger import logger
from util import timer
from time import time
from util import l2_loss, inner_product


class FastLightGCN(AbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(FastLightGCN, self).__init__(dataset, config)
        logger.info(config)
        # argument settings
        self.model_type = 'LightGCN'
        self.epoch = config["epoch"]
        self.adj_type = config["adj_type"]
        self.alg_type = config["alg_type"]
        self.n_users, self.n_items = dataset.num_users, dataset.num_items
        self.R = dataset.train_matrix
        self.dataset = dataset
        self.data_name = config["data.input.dataset"]
        self.n_fold = 100
        self.lr = config["learning_rate"]
        self.emb_dim = config["embed_size"]
        self.weight_size = config["weight_size"]
        self.node_dropout_flag = config["node_dropout_flag"]
        self.node_dropout = config["node_dropout"]
        self.mess_dropout = config["mess_dropout"]
        self.n_layers = len(self.weight_size)
        self.r_alpha = config["r_alpha"]
        self.fast_reg = config["fast_reg"]

        self.sess = sess

        plain_adj, norm_adj, mean_adj, pre_adj = self.get_adj_mat()

        if config["adj_type"] == 'plain':
            self.norm_adj = plain_adj
            print('use the plain adjacency matrix')
        elif config["adj_type"] == 'norm':
            self.norm_adj = norm_adj
            print('use the normalized adjacency matrix')
        elif config["adj_type"] == 'gcmc':
            self.norm_adj = mean_adj
            print('use the gcmc adjacency matrix')
        elif config["adj_type"] == 'pre':
            self.norm_adj = pre_adj
            print('use the pre adjcency matrix')
        else:
            config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        self.n_nonzero_elems = self.norm_adj.count_nonzero()

    def get_adj_mat(self):
        adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()

        # adj_mat = adj_mat
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()

        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        print('using xavier initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout_ph[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout_ph[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout_ph[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout_ph[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def _init_constant(self):
        # interaction information
        user_item_idx = [[u, i] for (u, i), r in self.dataset.train_matrix.todok().items()]
        user_idx, item_idx = list(zip(*user_item_idx))

        self.user_idx = tf.constant(user_idx, dtype=tf.int32, shape=None, name="user_idx")
        self.item_idx = tf.constant(item_idx, dtype=tf.int32, shape=None, name="item_idx")

    def build_graph(self):
        '''
                *********************************************************
                Create Placeholder for Input Data & Dropout.
                '''
        # placeholder definition
        self.users_ph = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items_ph = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items_ph = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).

        self.node_dropout_ph = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout_ph = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()
        self._init_constant()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        elif self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()


        """
        *********************************************************
        Inference for the testing phase.
        """
        # for prediction
        self.item_embeddings_final = tf.Variable(tf.zeros([self.n_items, self.emb_dim]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.n_users, self.emb_dim]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.ua_embeddings),
                           tf.assign(self.item_embeddings_final, self.ia_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.users_ph)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False,
                                       transpose_b=True)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.user_idx)
        self.i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.item_idx)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        # rating
        term1 = tf.matmul(self.ua_embeddings, self.ua_embeddings, transpose_a=True)
        term2 = tf.matmul(self.ia_embeddings, self.ia_embeddings, transpose_a=True)
        loss1 = tf.reduce_sum(tf.multiply(term1, term2))

        user_embed = tf.nn.embedding_lookup(self.ua_embeddings, self.user_idx)
        item_embed = tf.nn.embedding_lookup(self.ia_embeddings, self.item_idx)
        pos_ratings = inner_product(user_embed, item_embed)

        loss1 += tf.reduce_sum((self.r_alpha - 1) * tf.square(pos_ratings) - 2.0 * self.r_alpha * pos_ratings)
        # reg
        reg_loss = l2_loss(self.u_g_embeddings_pre, self.i_g_embeddings_pre)

        self.loss = loss1 + self.fast_reg * reg_loss

        # self.loss = self.mf_loss + self.emb_loss

        # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train_model(self):
        logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epoch):
            # _, _ = self.sess.run([self.update_opt, self.obj_loss])
            start = time()
            _, loss=self.sess.run([self.opt, self.loss])
            end = time()
            print('epoch%d: loss = %f time = %fs' % (epoch, loss, end - start))

            if epoch % 20 == 0:
                result = self.evaluate_model()
                logger.info("epoch %d:\t%s" % (epoch, result))

    @timer
    def evaluate_model(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, items=None):
        feed_dict = {self.users_ph: user_ids,
                     self.node_dropout_ph: [0.] * len(self.weight_size),
                     self.mess_dropout_ph: [0.] * len(self.weight_size)}
        i_rate_batch = self.sess.run(self.batch_ratings, feed_dict=feed_dict)

        return i_rate_batch
