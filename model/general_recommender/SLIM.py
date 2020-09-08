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
import xlrd
import xlwt
from xlutils.copy import copy

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
        y = self.train_mat[:, currentItem].toarray()
        if(currentItem % 10 == 0):
            logging.warning(currentItem)
        # set the j-th column of X to zero
        start_pos = self.train_mat.indptr[currentItem]
        end_pos = self.train_mat.indptr[currentItem + 1]

        current_item_data_backup = self.train_mat.data[start_pos: end_pos].copy()
        self.train_mat.data[start_pos: end_pos] = 0.0  # ensure wjj can be 0

        # fit one ElasticNet model per column
        self.model.fit(self.train_mat, y)

        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values

        # Select topK values
        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index

        nonzero_model_coef_index = self.model.sparse_coef_.indices
        nonzero_model_coef_value = self.model.sparse_coef_.data

        local_topK = min(len(nonzero_model_coef_value) - 1, self.Top_k_sparse)

        relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
        relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        self.lock.acquire()
        for index in range(len(ranking)):

            if self.numCells.value == len(self.rows):
                self.rows.extend([0]*self.dataBlock)
                self.cols.extend([0]*self.dataBlock)
                self.values.extend([0.0]*self.dataBlock)

            self.rows[self.numCells.value] = nonzero_model_coef_index[ranking[index]]
            self.cols[self.numCells.value] = currentItem
            self.values[self.numCells.value] = nonzero_model_coef_value[ranking[index]]

            self.numCells.value += 1
        self.lock.release()
        # finally, replace the original values of the j-th column
        self.train_mat.data[start_pos:end_pos] = current_item_data_backup


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
        pool = multiprocessing.Pool(5)
        # fit each item's factors sequentially (not in parallel)
        pool.map(self.do_col, range(self.num_items))
        pool.close()
        pool.join()
        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((self.values[:self.numCells.value], (self.rows[:self.numCells.value], self.cols[:self.numCells.value])),
                                       shape=(self.num_items, self.num_items), dtype=np.float32)
        self.ratings = self.train_mat.dot(self.W_sparse).toarray()

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        a = self.evaluate_val().split()
        oldWb = xlrd.open_workbook("SLIM.xls")
        oldWbS = oldWb.sheet_by_index(0)
        newWb = copy(oldWb)
        newWs = newWb.get_sheet(0)
        inserRowNo = oldWbS.nrows
        for colIndex in range(15):
            newWs.write(inserRowNo, colIndex, float(a[colIndex]))
        newWs.write(inserRowNo, 15, self.l1);
        newWs.write(inserRowNo, 16, self.l2);
        newWb.save('SLIM.xls')

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
            ratings = []
            for u, eval_items_by_u in zip(user_ids, candidate_items):
                ratings.append([self.ratings[u][i] for i in eval_items_by_u])
        return ratings

    def save_model(self):
        sps.save_npz(self.model_path + "_W.npz", self.W_sparse)


    def load_model(self):
        self.W_sparse = sps.load_npz(self.model_path + "_W.npz")
        self.ratings = self.train_mat.dot(self.W_sparse).toarray()
