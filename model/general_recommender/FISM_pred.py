"""
Reference: Santosh Kabbur et al., "FISM: Factored Item Similarity Models for Top-N Recommender Systems." in KDD 2013.
@author: wubin
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner, data_generator, tool
from util import timer
from util.tool import csr_to_user_dict
from util import l2_loss
from util.data_iterator import DataIterator
from util import pad_sequences
import pickle


class FISM_pred(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(FISM_pred, self).__init__(dataset, conf)
        self.batch_size = conf["batch_size"]
        self.num_epochs = conf["epochs"]
        self.embedding_size = conf["embedding_size"]
        self.lambda_bilinear = conf["lambda"]
        self.gamma_bilinear = conf["gamma"]
        self.alpha = conf["alpha"]
        self.num_negatives = conf["num_neg"]
        self.learning_rate = conf["learning_rate"]
        self.learner = conf["learner"]
        self.topK = conf["topk"]
        self.loss_function = conf["loss_function"]
        self.is_pairwise = conf["is_pairwise"]
        self.num_negatives = conf["num_neg"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.verbose = conf["verbose"]
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.dataset = dataset
        self.pretrain_file = "D:\\data\\NeuRec\\%s_epoch=%d_fism.pkl" % (str(self.gamma_bilinear).replace('.', '_') + str(self.lambda_bilinear).replace('.', '_'), self.num_epochs)
        self.train_dict = csr_to_user_dict(self.dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None], name="user_input")  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None],
                                          name="num_idx")  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input_pos")  # the index of items
            if self.is_pairwise is True:
                self.user_input_neg = tf.placeholder(tf.int32, shape=[None, None], name="user_input_neg")
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
                self.num_idx_neg = tf.placeholder(tf.float32, shape=[None], name="num_idx_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

    def _create_variables(self, params):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.c1 = tf.Variable(tf.constant(params[0]), name='c1', dtype=tf.float32)
            self.embedding_Q = tf.Variable(tf.constant(params[1]), name='embedding_Q', dtype=tf.float32)
            self.bias = tf.Variable(tf.constant(params[2]), name="bias", dtype=tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], 0, name='embedding_Q_')

    def _create_inference(self, user_input, item_input, num_idx):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q_, user_input), 1)
            embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            output = coeff * tf.reduce_sum(tf.multiply(embedding_p, embedding_q), 1) + bias_i
        return embedding_p, embedding_q, output

    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, self.output = self._create_inference(self.user_input, self.item_input, self.num_idx)
            if self.is_pairwise is True:
                _, q2, output_neg = self._create_inference(self.user_input_neg, self.item_input_neg, self.num_idx_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function, self.result) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q2, q1)

            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q1)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)

    def build_graph(self):
        params = []
        with open(self.pretrain_file, "rb") as fin:
            params.extend(pickle.load(fin, encoding="utf-8"))
        self._create_placeholders()
        self._create_variables(params)
        self._create_loss()
        self._create_optimizer()

    def train_model(self):
        self.logger.info(self.evaluate_val())

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    @timer
    def evaluate_val(self):
        return self.evaluator_val.evaluate(self)

    def predict(self, user_ids, candidate_items_userids):
        ratings = []
        if candidate_items_userids is not None:
            for u, eval_items_by_u in zip(user_ids, candidate_items_userids):
                user_input = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                item_idx = np.full(len(eval_items_by_u), num_idx, dtype=np.int32)
                user_input.extend([cand_items_by_u] * len(eval_items_by_u))
                feed_dict = {self.user_input: user_input,
                             self.num_idx: item_idx,
                             self.item_input: eval_items_by_u}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))

        else:
            eval_items = np.arange(self.num_items)
            for u in user_ids:
                user_input = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                item_idx = np.full(self.num_items, num_idx, dtype=np.int32)
                user_input.extend([cand_items_by_u] * self.num_items)
                ratings_row = []
                if (True):
                    print("u_id:" + str(u))
                for i in range(int(self.num_items / 500) + 1):
                    feed_dict = {self.user_input: user_input[i * 500: i * 500 + 500],
                                 self.num_idx: item_idx[i * 500: i * 500 + 500],
                                 self.item_input: eval_items[i * 500: i * 500 + 500]}
                    ratings_row.extend(self.sess.run(self.output, feed_dict=feed_dict))
                ratings.append(ratings_row)
        return ratings
