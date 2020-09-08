#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@author: Jiarui Yu
reference: https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
"""
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps
import numpy as np
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
from util import timer
import xlwt
import xlrd
from xlutils.copy import copy
class pureSVD(AbstractRecommender):
    """ PureSVDRecommender
    DOI:https://doi.org/10.1145/1864708.1864721
    """
    def __init__(self, sess, dataset, conf):
        super(pureSVD, self).__init__(dataset, conf)
        self.verbose = conf['verbose']
        self.num_factors = conf['num_factors']
        self.hyperparameter_list = [str(self.num_factors)]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.ITEM_factors = None
        self.USER_factors = None
        self.sess = sess
        self.model_path = "/ghome/yujr/saved_models/pureSVD" + "/" + "_".join(
            ["\"" + hyperpara + "\"" for hyperpara in self.hyperparameter_list])

    def build_graph(self, random_seed=None):
        print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.dataset.train_matrix,
                                      n_components=self.num_factors,
                                      random_state=random_seed)

        U_s = U * sps.diags(Sigma)

        self.USER_factors = U_s
        self.ITEM_factors = VT

        print("Computing SVD decomposition... Done!")

    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    @timer
    def evaluate_val(self):
        return self.evaluator_val.evaluate(self)

    def train_model(self, random_seed = None):
        self.logger.info(self.evaluator.metrics_info())
        self.ratings_matrix = self.USER_factors.dot(self.ITEM_factors)
        a = self.evaluate_val().split()
        oldWb = xlrd.open_workbook("PureSVD.xls", formatting_info=True);
        oldWbS = oldWb.sheet_by_index(0)
        newWb = copy(oldWb)
        newWs = newWb.get_sheet(0)
        inserRowNo = oldWbS.nrows
        for colIndex in range(15):
            newWs.write(inserRowNo, colIndex, float(a[colIndex]));
        newWs.write(inserRowNo, 15, self.num_factors);
        newWb.save('PureSVD.xls')

    def predict(self, user_ids, candidate_items):
        if candidate_items is None:
            return self.ratings_matrix[user_ids]
        else:
            ratings = []
            for u, eval_items_by_u in zip(user_ids, candidate_items):
                ratings.append([self.ratings_matrix[u][i] for i in eval_items_by_u])
        return ratings

    def save_model(self):
        np.savetxt(self.model_path + "_item_factors.txt", self.ITEM_factors)
        np.savetxt(self.model_path + "_user_factors.txt", self.USER_factors)

    def load_model(self):
        try:
            self.ITEM_factors = np.loadtxt(self.model_path + "_item_factors.txt")
            self.USER_factors = np.loadtxt(self.model_path + "_user_factors.txt")
        except:
            print('no such model')