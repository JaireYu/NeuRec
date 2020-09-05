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
        pass

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            ratings = self.USER_factors.dot(self.ITEM_factors)
            return ratings[user_ids]
        else:
            ratings = None
            """waiting to complete"""
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