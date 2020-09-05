"""
Reference: Xiangnan He et al., "NAIS: Neural Attentive Item Similarity Model for Recommendation." in TKDE2018
@author: wubin
"""
from model.AbstractRecommender import AbstractRecommender
import tensorflow as tf
import numpy as np
from time import time
from util import learner,data_generator, tool
from util import timer
from util.tool import csr_to_user_dict
import pickle
from util import l2_loss
from util import pad_sequences
from util.data_iterator import DataIterator
import multiprocessing
import itertools
class NAIS(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(NAIS, self).__init__(dataset, conf)
        self.pretrain = conf["pretrain"]
        self.verbose = conf["verbose"]
        self.batch_size = conf["batch_size"]
        self.num_epochs = conf["epochs"]
        self.weight_size = conf["weight_size"]
        self.embedding_size = conf["embedding_size"]
        self.is_pairwise = conf["is_pairwise"]
        self.topK = conf["topk"]
        self.lambda_bilinear = conf["lambda"]
        self.gamma_bilinear = conf["gamma"]
        self.W_reg = conf["W_reg"]
        self.H_reg = conf["H_reg"]
        self.alpha = conf["alpha"]
        self.beta = conf["beta"]
        self.num_negatives = conf["num_neg"]
        self.learning_rate = conf["learning_rate"]
        self.activation = conf["activation"]
        self.loss_function = conf["loss_function"]
        self.algorithm = conf["algorithm"]
        self.learner = conf["learner"]
        self.embed_init_method = conf["embed_init_method"]
        self.weight_init_method = conf["weight_init_method"]
        self.stddev = conf["stddev"]
        self.pretrain_file = conf["pretrain_file"]
        self.dataset = dataset
        self.num_items = dataset.num_items
        self.num_users = dataset.num_users
        self.train_dict = csr_to_user_dict(self.dataset.train_matrix)
        self.pred_g = tf.Graph()
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None], name="user_input")  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None], name="num_idx")  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input_pos")  # the index of items
            if self.is_pairwise is True:
                self.user_input_neg = tf.placeholder(tf.int32, shape=[None, None], name="user_input_neg")
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
                self.num_idx_neg = tf.placeholder(tf.float32, shape=[None], name="num_idx_neg")
            else:
                self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

    def _create_variables(self, params=None):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            if params is None:
                #使用init的方法init
                embed_initializer = tool.get_initializer(self.embed_init_method, self.stddev)
                
                self.c1 = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                      name='c1', dtype=tf.float32)
                self.embedding_Q = tf.Variable(embed_initializer([self.num_items, self.embedding_size]),
                                               name='embedding_Q', dtype=tf.float32)
                self.bias = tf.Variable(tf.zeros(self.num_items), name='bias')
            else:
                #使用预训练进行初始化
                self.c1 = tf.Variable(np.array(params[0][0]), name='c1', dtype=tf.float32)
                self.embedding_Q = tf.Variable(np.array(params[0][1]), name='embedding_Q', dtype=tf.float32)
                self.bias = tf.Variable(np.array(params[0][2]), name="bias", dtype=tf.float32)
                
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], axis=0, name='embedding_Q_')

            # Variables for attention
            weight_initializer = tool.get_initializer(self.weight_init_method, self.stddev)
            #计算相似度的两种算法
            if self.algorithm == 0:
                self.W = tf.Variable(weight_initializer([self.embedding_size, self.weight_size]),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:    
                self.W = tf.Variable(weight_initializer([2*self.embedding_size, self.weight_size]),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            
            self.b = tf.Variable(weight_initializer([1, self.weight_size]),
                                 name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)
            
    def _create_inference(self, user_input, item_input, num_idx):
        with tf.name_scope("inference"):
            embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, user_input)  # (b, n, e)
            embedding_q = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, 1, e)
            
            if self.algorithm == 0:
                embedding_p = self._attention_mlp(embedding_q_ * embedding_q, embedding_q_, num_idx)
            else:
                n = tf.shape(user_input)[1]
                # sum(a_{ij} q_j)
                embedding_p = self._attention_mlp(tf.concat([embedding_q_, tf.tile(embedding_q, tf.stack([1, n, 1]))], 2),
                                                  embedding_q_, num_idx)

            embedding_q = tf.reduce_sum(embedding_q, 1)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, tf.constant(self.alpha, tf.float32, [1]))
            output = coeff * tf.reduce_sum(embedding_p*embedding_q, 1) + bias_i
            
            return embedding_q_, embedding_q, output
    
    def _create_loss(self):
        with tf.name_scope("loss"):
            p1, q1, self.output = self._create_inference(self.user_input,self.item_input,self.num_idx)
            if self.is_pairwise is True:
                _, q2, output_neg = self._create_inference(self.user_input_neg, self.item_input_neg, self.num_idx_neg)
                self.result = self.output - output_neg
                self.loss = learner.pairwise_loss(self.loss_function, self.result) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q2, q1)
            
            else:
                self.loss = learner.pointwise_loss(self.loss_function, self.labels, self.output) + \
                            self.lambda_bilinear * l2_loss(p1) + \
                            self.gamma_bilinear * l2_loss(q1) + self.W_reg * l2_loss(self.W) +\
                            self.H_reg * l2_loss(self.h)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
            
    def build_graph(self):
        self._create_placeholders()
        try:
            pre_trained_params = []
            with open(self.pretrain_file, "rb") as fin:
                pre_trained_params.append(pickle.load(fin, encoding="utf-8"))
            #with open(self.mlp_pretrain, "rb") as fin:
            #    pre_trained_params.append(pickle.load(fin, encoding="utf-8"))
            self.logger.info("load pretrained params successful!")
        except:
            pre_trained_params = None
            self.logger.info("load pretrained params unsuccessful!")
            
        self._create_variables(pre_trained_params)
        self._create_loss()
        self._create_optimizer()
        """
        with tf.name_scope('some_scope1'):
            self.embedding_q_ = tf.placeholder(tf.float32, shape=[None, None], name="q_embedding")  # iteraction, 64
            self.embedding_q = tf.placeholder(tf.float32, shape=[None, None],
                                         name="p_embedding")  # item, 64
            self.cur_bias = tf.placeholder(tf.float32, shape=[None], name="bias")
            self.cur_W = tf.placeholder(tf.float32, shape=[None, None], name="bias")
            self.cur_b = tf.placeholder(tf.float32, shape=[None], name="bias")
            embedding_q = tf.expand_dims(self.embedding_q, 1)  # 100*1*64
            q_ = self.embedding_q_ * embedding_q  # 100*159*64
            b = np.shape(q_)[0]  # 100
            n = np.shape(q_)[1]  # 159
            r = (self.algorithm + 1) * 64  # 64
            mlp_output = tf.matmul(tf.reshape(q_, [-1, r]),  # (100*159)*64
                                   self.cur_W) + self.cur_b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                mlp_output = tf.nn.relu(mlp_output)
            elif self.activation == 1:
                mlp_output = tf.nn.sigmoid(mlp_output)
            elif self.activation == 2:
                mlp_output = tf.nn.tanh(mlp_output)

            # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(tf.reshape(tf.matmul(mlp_output, self.cur_h), [b, n]))  # 100*159
            # mask_mat 标志每个用户的商品，为了计算不同长度的sum
            exp_sum = tf.sum(exp_A_, axis=1, keepdims=True)  # 100*1
            exp_sum = tf.pow(exp_sum, self.beta)  # 100*1

            exp_A_ = tf.expand_dims(tf.divide(exp_A_, exp_sum), 2)  # 100*159*1
            embedding_q = tf.sum(embedding_q, 1)  # 100*64
            embedding_p_ = tf.sum(exp_A_ * self.embedding_q_, 1)  # 100*64
            output = tf.sum(embedding_p_ * embedding_q, 1) + self.cur_bias  # 100
        """

    def _attention_mlp(self, q_, embedding_q_, num_idx):
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = (self.algorithm + 1)*self.embedding_size

            mlp_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                mlp_output = tf.nn.relu(mlp_output)
            elif self.activation == 1:
                mlp_output = tf.nn.sigmoid(mlp_output)
            elif self.activation == 2:
                mlp_output = tf.nn.tanh(mlp_output)

            A_ = tf.reshape(tf.matmul(mlp_output, self.h), [b,n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            #mask_mat 标志每个用户的商品，为了计算不同长度的sum
            mask_mat = tf.sequence_mask(num_idx, maxlen = n, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return tf.reduce_sum(A * embedding_q_, 1)

    def _attention_mlp_pred(self, q_, embedding_q_, batch_size, itemj_num):
        with tf.name_scope("attention_MLP"):
            b = np.shape(q_)[1]
            n = np.shape(q_)[2]
            r = (self.algorithm + 1) * self.embedding_size
            mask_mat = np.zeros((batch_size, n))
            for index, i in enumerate(itemj_num):
                mask_mat[index][0:i] = 1
            np.expand_dims(mask_mat, 1)
            mlp_output = np.matmul(np.reshape(q_, [-1, r]), self.cur_W) + self.cur_b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                mlp_output = np.maximum(0, mlp_output)
            elif self.activation == 1:
                mlp_output = 1 / (1 + np.exp(-mlp_output))
            elif self.activation == 2:
                mlp_output = np.tanh(mlp_output)

            A_ = np.reshape(np.matmul(mlp_output, self.cur_h), [batch_size, b, n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = mask_mat *np.exp(A_)
            # mask_mat 标志每个用户的商品，为了计算不同长度的sum
            exp_sum = np.sum(exp_A_, axis=2, keepdims=True)  # (b, 1)
            exp_sum = np.power(exp_sum, self.beta)

            A = np.expand_dims(np.divide(exp_A_, exp_sum), 3)  # (b, n, 1)

            return np.sum(A * embedding_q_, 1)

    def train_model(self):
        #self.logger.info("epoch %d:\t%s" % (0, self.evaluate()))
        for epoch in range(1, self.num_epochs+1):
            if self.is_pairwise is True:
                user_input, user_input_neg, num_idx_pos, num_idx_neg, item_input_pos, item_input_neg = \
                    data_generator._get_pairwise_all_likefism_data(self.dataset)
                data_iter = DataIterator(user_input, user_input_neg, num_idx_pos,
                                         num_idx_neg, item_input_pos, item_input_neg,
                                         batch_size=self.batch_size, shuffle=True)
            else:
                user_input, num_idx, item_input, labels, batch_length = \
                 data_generator._get_pointwise_all_likefism_data_debug_fast(self.dataset, self.num_negatives, self.train_dict)
                #data_iter = DataIterator(user_input, num_idx, item_input, labels,
                #                         batch_size=self.batch_size, shuffle=False)
            num_training_instances = len(user_input)
            total_loss = 0.0
            training_start_time = time()
            if self.is_pairwise is True:
                for bat_users_pos, bat_users_neg, bat_idx_pos, bat_idx_neg, bat_items_pos, bat_items_neg in data_iter:
                    bat_users_pos = pad_sequences(bat_users_pos, value=self.num_items)
                    bat_users_neg = pad_sequences(bat_users_neg, value=self.num_items)
                    feed_dict = {self.user_input: bat_users_pos,
                                 self.user_input_neg: bat_users_neg,
                                 self.num_idx: bat_idx_pos,
                                 self.num_idx_neg: bat_idx_neg,
                                 self.item_input: bat_items_pos,
                                 self.item_input_neg: bat_items_neg}

                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss
            else:
                for index in range(len(batch_length)-1):
                    temp = pad_sequences(user_input[batch_length[index]:batch_length[index+1]], value=self.num_items)
                    feed_dict = {self.user_input: temp,
                                 self.num_idx: num_idx[batch_length[index]:batch_length[index+1]],
                                 self.item_input: item_input[batch_length[index]:batch_length[index+1]],
                                 self.labels: labels[batch_length[index]:batch_length[index+1]]}
                    loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                    total_loss += loss

            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/num_training_instances,
                                                             time()-training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate_val()))
        
        # save model
        # params = self.sess.run([self.c1, self.embedding_Q, self.bias])
        # with open("./pretrained/%s_epoch=%d_fism.pkl" % (self.dataset_name, self.num_epochs), "wb") as fout:
        #     pickle.dump(params, fout)
    @timer
    def evaluate(self):
        #self.cur_embedding_Q_, self.cur_embedding_Q, self.cur_bias, self.cur_W, self.cur_b, self.cur_h \
        #    = self.sess.run([self.embedding_Q_, self.embedding_Q, self.bias, self.W, self.b, self.h])
        item_batch_size = 1000
        self.ratings = np.empty((self.num_users, self.num_items))
        eval_items = np.arange(self.num_items)
        indices = []
        for i in range(self.num_users):
            indices.append(len(self.train_dict[i]))
        indices = np.argsort(np.array(indices))
        """
        u_ids = []
        for u in indices:
            u_ids.extend([u]*self.num_items)
        u_ids = np.array(u_ids)
        i_ids = np.tile(eval_items, self.num_users)
        user_inputs = []
        for u in indices:
            item_by_user = self.train_dict[u]
            for i in i_ids:
                user_inputs.append(item_by_user)
        """
        batch_length = [0]
        last_point = 0
        cnt = 0
        shrehold = 40
        global_user_cnt = 0
        for _i_, u in enumerate(indices):
           if (cnt - last_point+1) * len(self.train_dict[u]) > shrehold:
               batch_length.append(cnt)
               last_point = cnt
           cnt += 1
        batch_length.append(self.num_users)
        for index in range(len(batch_length)-1):
            if batch_length[index+1]-batch_length[index]>1:
                user_input = []
                item_idx = np.array([])
                for u in indices[batch_length[index]:batch_length[index+1]]:
                    cand_items_by_u = self.train_dict[u]
                    num_idx = len(cand_items_by_u)
                    item_idx = np.append(item_idx, np.full(self.num_items, num_idx, dtype=np.int32))
                    user_input.extend([cand_items_by_u] * self.num_items)
                user_input = pad_sequences(user_input, value=self.num_items)
                feed_dict = {self.user_input: user_input,
                             self.num_idx: item_idx,
                             self.item_input: np.tile(eval_items, batch_length[index+1]-batch_length[index])}
                temp_data = np.reshape(self.sess.run(self.output, feed_dict=feed_dict), (-1, self.num_items))
                for i, u in enumerate(indices[batch_length[index]:batch_length[index+1]]):
                    self.ratings[u] = temp_data[i]
            else:
                u = indices[batch_length[index]]
                ratings_row = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                #item_batch_size = 2000000 // num_idx
                item_batch_size = 4000000 // num_idx
                item_batch = self.num_items // item_batch_size + 1
                for item in range(item_batch):
                    start = item * item_batch_size
                    end = min((item + 1) * item_batch_size, self.num_items)
                    user_input = []
                    item_idx = np.full(end - start, num_idx, dtype=np.int32)
                    user_input.extend([cand_items_by_u] * (end - start))
                    feed_dict = {self.user_input: user_input,
                                 self.num_idx: item_idx,
                                 self.item_input: eval_items[start:end]}
                    ratings_row.extend(self.sess.run(self.output, feed_dict=feed_dict))
                self.ratings[u] = np.array(ratings_row)
        return self.evaluator.evaluate(self)

    @timer
    def evaluate_val(self):
        #self.cur_embedding_Q_, self.cur_embedding_Q, self.cur_bias, self.cur_W, self.cur_b, self.cur_h \
        #    = self.sess.run([self.embedding_Q_, self.embedding_Q, self.bias, self.W, self.b, self.h])
        return self.evaluator_val.evaluate(self)


    def predict(self, user_ids, candidate_items_userids):
        ratings = []
        if candidate_items_userids is not None:
            """
            embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, user_input)  # (b, n, e)
            embedding_q = tf.expand_dims(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, 1, e)

            if self.algorithm == 0:
                embedding_p = self._attention_mlp(embedding_q_ * embedding_q, embedding_q_, num_idx)
            else:
                n = tf.shape(user_input)[1]
                # sum(a_{ij} q_j)
                embedding_p = self._attention_mlp(
                    tf.concat([embedding_q_, tf.tile(embedding_q, tf.stack([1, n, 1]))], 2),
                    embedding_q_, num_idx)

            embedding_q = tf.reduce_sum(embedding_q, 1)
            bias_i = tf.nn.embedding_lookup(self.bias, item_input)
            coeff = tf.pow(num_idx, tf.constant(self.alpha, tf.float32, [1]))
            output = coeff * tf.reduce_sum(embedding_p * embedding_q, 1) + bias_i

            return embedding_q_, embedding_q, output
            """
        else:
            u_start = user_ids[0]
            return self.ratings[u_start:u_start+len(user_ids)]
            """
            item_batch_size = 1000
            ratings = []
            eval_items = np.arange(self.num_items)
            for u in user_ids:
                print(u)
                ratings_row = []
                cand_items_by_u = self.train_dict[u]
                num_idx = len(cand_items_by_u)
                item_batch_size = 2000000//num_idx
                item_batch = self.num_items // item_batch_size + 1
                for item in range(item_batch):
                    start = item * item_batch_size
                    end = min((item + 1) * item_batch_size, self.num_items)
                    user_input = []
                    item_idx = np.full(end-start, num_idx, dtype=np.int32)
                    user_input.extend([cand_items_by_u] * (end-start))
                    feed_dict = {self.user_input: user_input,
                                 self.num_idx: item_idx,
                                 self.item_input: eval_items[start:end]}
                    ratings_row.extend(self.sess.run(self.output, feed_dict=feed_dict))
                ratings.append(ratings_row)
            """
            """
            item_batch_size = 100
            for u in user_ids:
                print(u)
                cand_items_by_u = np.array(self.train_dict[u])  # 159
                # item_idx = np.full(self.num_items, num_idx, dtype=np.int32)
                # user_input.extend([cand_items_by_u]*self.num_items)
                embedding_q_ = self.cur_embedding_Q_[cand_items_by_u]  # 159*64
                num_idx = len(cand_items_by_u)
                item_batch_size = 2000000//num_idx
                item_batch = self.num_items // item_batch_size + 1
                ratings_row = []
                for item in range(item_batch):
                    start = item * item_batch_size
                    end = min((item+1)*item_batch_size, self.num_items)
                    embedding_q = np.expand_dims(self.cur_embedding_Q[start:end, :], 1)  # 100*1*64
                    q_ = embedding_q_ * embedding_q # 100*159*64
                    b = np.shape(q_)[0] #100
                    n = np.shape(q_)[1] #159
                    r = (self.algorithm + 1) * 64 #64
                    mlp_output = np.matmul(np.reshape(q_, [-1, r]), # (100*159)*64
                                           self.cur_W) + self.cur_b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
                    if self.activation == 0:
                        mlp_output = np.maximum(0, mlp_output)
                    elif self.activation == 1:
                        mlp_output = 1 / (1 + np.exp(-mlp_output))
                    elif self.activation == 2:
                        mlp_output = np.tanh(mlp_output)

                    # (b*n, w) * (w, 1) => (None, 1) => (b, n)

                    # softmax for not mask features
                    exp_A_ = np.exp(np.reshape(np.matmul(mlp_output, self.cur_h), [b, n])) # 100*159
                    # mask_mat 标志每个用户的商品，为了计算不同长度的sum
                    exp_sum = np.sum(exp_A_, axis=1, keepdims=True)  # 100*1
                    exp_sum = np.power(exp_sum, self.beta)           # 100*1

                    exp_A_ = np.expand_dims(np.divide(exp_A_, exp_sum), 2)  # 100*159*1
                    embedding_q = np.sum(embedding_q, 1)                    # 100*64
                    embedding_p_ = np.sum(exp_A_ * embedding_q_, 1)         # 100*64
                    output = np.sum(embedding_p_ * embedding_q, 1) + self.cur_bias[start:end] #100
                    ratings_row.extend(output)
                ratings.append(ratings_row)
            """
            """
            return pred(np.array(user_ids), self.train_dict, self.cur_embedding_Q_
                        , self.cur_embedding_Q, self.cur_bias, self.cur_W
                        , self.cur_h, self.cur_b, self.num_users, self.num_items
                        , self.algorithm, self.activation, self.beta)
            """
        return ratings
