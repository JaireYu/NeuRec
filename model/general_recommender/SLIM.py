#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@author: Jiarui Yu
reference: https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
reference: https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
"""
import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from model.AbstractRecommender import AbstractRecommender
from sklearn.exceptions import ConvergenceWarning
from util import timer
import warnings
import multiprocessing
import os
import logging

class SLIM(AbstractRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self, sess, dataset, conf):
        super(SLIM, self).__init__(dataset, conf)
        self.verbose = conf['verbose']
        self.Top_k_sparse = conf['Top_k_sparse']
        self.l1 = conf['l1']
        self.l2 = conf['l2']
        self.positive_only = conf['positive_only']
        self.hyperparameter_list = [str(self.Top_k_sparse), str(self.l1), str(self.l2)]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.model_path = "/ghome/yujr/NeuRec-dev/"+"_".join(
            [hyperpara for hyperpara in self.hyperparameter_list])
        self.lock = None
        self.rows = None
        self.cols = None
        self.values = None
        self.train_mat = self.dataset.train_matrix.tocsc()
        self.numCells = None

    def do_col(self, currentItem):
        print(currentItem)

    def build_graph(self):
        l1_ratio = self.l1/(2*self.l2 + self.l1)
        alpha = 2*self.l2 + self.l1
        assert l1_ratio>= 0 and l1_ratio<=1, "parameter error"
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        # Display ConvergenceWarning only once and not for every item it occurs
        warnings.simplefilter("once", category = ConvergenceWarning)

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=self.alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                #max_iter=100,
                                tol=1e-4)

        # Use array as it reduces memory requirements compared to lists

        self.dataBlock = 10000000
        manager = multiprocessing.Manager()
        self.rows = manager.list([0] * self.dataBlock)
        self.cols = manager.list([0] * self.dataBlock)
        self.values = manager.list([0.0] * self.dataBlock)
        self.numCells = manager.Value('i', 0)
        self.lock = manager.Lock()
        pool = multiprocessing.Pool(8)
        # fit each item's factors sequentially (not in parallel)
        pool.map(self.do_col, range(10))
        pool.close()
        pool.join()
        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((self.values[:self.numCells.value], (self.rows[:self.numCells.value], self.cols[:self.numCells.value])),
                                       shape=(self.num_items, self.num_items), dtype=np.float32)
        self.ratings = self.train_mat.dot(self.W_sparse).toarray()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    @timer
    def evaluate_val(self):
        return self.evaluator_val.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            return self.ratings[user_ids]
        else:
            ratings = None
            """waiting to complete"""
        return ratings

    def save_model(self):
        sps.save_npz(self.model_path + "_W.npz", self.W_sparse)


    def load_model(self):
        self.W_sparse = sps.load_npz(self.model_path + "_W.npz")
        self.ratings = self.train_mat.dot(self.W_sparse).toarray()
