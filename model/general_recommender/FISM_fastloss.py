#!/usr/local/bin/python
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
from util.tool import inner_product


class FISM_fastloss(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(FISM_fastloss, self).__init__(dataset, conf)
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
        self.r_alpha = conf["r_alpha"]
        self.train_dict = csr_to_user_dict(self.dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.convert_matrix = tf.placeholder(tf.float32, shape = (self.num_users, self.num_items))

    def _create_constant(self):
        with tf.name_scope("convert_mat"):
            self.num_idx = []
            for u in range(self.num_users):
                self.num_idx.append(len(self.train_dict[u]))
            self.num_idx = np.power(np.array(self.num_idx), -self.alpha)
            self.coef = tf.constant(self.num_idx, tf.float32, [self.num_users, 1], name='coef')
            user_item_idx = [[u, i] for (u, i), r in self.dataset.train_matrix.todok().items()]
            user_idx, item_idx = list(zip(*user_item_idx))

            self.user_idx = tf.constant(user_idx, dtype=tf.int32, shape=None, name="user_idx")
            self.item_idx = tf.constant(item_idx, dtype=tf.int32, shape=None, name="item_idx")

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            initializer = tool.get_initializer(self.init_method, self.stddev)
            self.embedding_Q_ = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                  name='embedding_Q_', dtype=tf.float32)
            self.item_embeddings= tf.Variable(initializer([self.num_items, self.embedding_size]),
                                           name='embedding_Q', dtype=tf.float32)

    def _create_user_embeddings(self):
        with tf.name_scope("user_embedding"):
            self.user_embeddings = tf.multiply(tf.matmul(self.convert_matrix, self.embedding_Q_), self.coef)


    def _create_inference(self, user_input):
        with tf.name_scope("inference"):
            embedding_p = tf.expand_dims(tf.nn.embedding_lookup(self.user_embeddings, user_input), 1)
            self.output = tf.multiply(embedding_p, self.embedding_Q)

    def _create_loss(self):
        with tf.name_scope("loss"):
            term1 = tf.matmul(tf.transpose(self.user_embeddings), self.user_embeddings)
            term2 = tf.matmul(tf.transpose(self.item_embeddings), self.item_embeddings)
            loss1 = tf.reduce_sum(tf.multiply(term1, term2))

            user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx)
            item_embed = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx)
            pos_ratings = inner_product(user_embed, item_embed)

            loss1 += tf.reduce_sum((self.r_alpha - 1) * tf.square(pos_ratings) - 2.0 * self.r_alpha * pos_ratings)
            # reg
            self.loss = loss1 + \
                        self.lambda_bilinear * l2_loss(self.embedding_Q_) + \
                        self.gamma_bilinear * l2_loss(self.item_embeddings)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)

    def build_graph(self):
        self.interacts = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        for u in range(self.num_users):
            for i in self.train_dict[u]:
                self.interacts[u][i] = 1
        self._create_placeholders()
        self._create_constant()
        self._create_variables()
        self._create_user_embeddings()
        self._create_loss()
        self._create_optimizer()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0.0
            training_start_time = time()
            loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict={self.convert_matrix: self.interacts})

            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, loss,
                                                                  time() - training_start_time))
            if epoch % 1 == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate_val()))
            #if epoch % self.verbose == 0:
            #    self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate_val()))
            #    self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        self.cur_user_embeddings, self.cur_item_embeddings \
            = self.sess.run([self.user_embeddings, self.item_embeddings], feed_dict={self.convert_matrix: self.interacts})
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
                #ratings.append(self.sess.run(self.output, feed_dict=feed_dict))

        else:
            ratings = np.multiply\
                (np.expand_dims
                 (self.cur_user_embeddings[user_ids[0]:user_ids[0]+len(user_ids)], 1),
                                                   self.cur_item_embeddings)
            return ratings
        return ratings