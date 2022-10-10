import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.nn.rnn_cell import GRUCell


print('Model-VERSION:', 1400)

class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len, flag="DNN", l2_reg=0,
                 boost_ratio=1, kernel_type='splus', weight_type='even', unit=True):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.dim = embedding_dim
        self.l2_reg = l2_reg
        self.boost_ratio = boost_ratio # 1
        self.kernel_type = kernel_type # splus
        self.weight_type = weight_type # even
        self.unit = bool(unit)
        if self.l2_reg > 0:
            print('Use l2 regularization, rate:', self.l2_reg)
        self.neg_num = neg_num
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.neg_sample_ph = tf.placeholder(tf.int32, [None, ], name='neg_sample_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.mid_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(), trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

    def build_sampled_pt_loss(self, item_emb, user_emb):
        if self.unit:
            item_norm = tf.reduce_sum(tf.square(item_emb), -1) + 1e-8
            item_emb = item_emb / tf.expand_dims(tf.sqrt(item_norm), -1)
            item_norm_new = tf.reduce_sum(tf.square(item_emb), -1) + 1e-8
            user_norm = tf.reduce_sum(tf.square(user_emb), -1) + 1e-8
            user_emb = user_emb / tf.expand_dims(tf.sqrt(user_norm), -1)
        else:
            # for print
            item_norm = tf.reduce_sum(tf.square(item_emb), -1) + 1e-8
            item_norm_new = item_norm

        self.check_rkpct(item_emb, user_emb)
        print('----- pt loss HyperP setting:', self.boost_ratio, self.kernel_type, self.weight_type, self.neg_num * self.batch_size, self.unit)

        neg_samples = self.neg_sample_ph
        neg_embs = tf.nn.embedding_lookup(self.mid_embeddings_var, neg_samples) # (neg_num, dim)
        if self.unit:
            neg_norm = tf.reduce_sum(tf.square(neg_embs), -1) + 1e-8
            neg_embs = neg_embs / tf.expand_dims(tf.sqrt(neg_norm), -1)

        pos_logit = tf.reduce_sum(user_emb * item_emb, -1, keepdims=True) # (b, 1)
        neg_logit = tf.matmul(user_emb, neg_embs, transpose_b=True) # (b, n)
        pair_diff = pos_logit - neg_logit # (b, n)
        self.pair_located = -0.5 * pair_diff

        kernel_dict = {
            'exp': lambda x: tf.exp(x),
            'splus': lambda x: tf.log(1 + tf.exp(x)), # try log(1+exp(2x))/log(2)
            'hinge': lambda x: tf.nn.relu(x + 0.5),
            'hinge5': lambda x: tf.nn.relu(x + 5),
            'hinge5z': lambda x: 0.1 * tf.nn.relu(x + 5) + tf.nn.relu(x),
            'sigmoid': lambda x: tf.nn.sigmoid(x),
            'signz': lambda x: 5 * tf.nn.relu(x + 0.1) + (0.5 / tf.stop_gradient(tf.reduce_max(x, -1, keepdims=True)+1e-5) - 5) * tf.nn.relu(x),
            'sighinge': lambda x: tf.sigmoid(-tf.nn.relu(-x)) + tf.nn.relu(x / 20),
            'sigsigg': lambda x: tf.sigmoid(-tf.nn.relu(-x)) + tf.sigmoid(tf.nn.relu(0.5 * x)) - 0.5,
            'siggsig': lambda x: tf.sigmoid(-tf.nn.relu(-0.5 * x)) + tf.sigmoid(tf.nn.relu(x)) - 0.5,
        }

        rk_i = tf.reduce_sum(kernel_dict[self.kernel_type](pair_diff * -self.boost_ratio), -1) + 1 # [b, ]
        hard_rk_i = tf.reduce_sum(tf.nn.relu(tf.sign(-pair_diff)), -1) + 1

        # neg_sample_num = tf.cast(self.neg_num * self.batch_size, tf.float32)
        N = tf.cast(self.n_mid, dtype=tf.float32)
        M = self.neg_num * self.batch_size
        R = N / M

        rkpct_i = rk_i / (M / 100)
        hard_rkpct_i = hard_rk_i / (M / 100)
        distribution_dict = {
            # s for sharp, f for flat
            'distexps': tf.distributions.Exponential(rate=0.1),
            'distexpss': tf.distributions.Exponential(rate=0.2),
            'distexpsss': tf.distributions.Exponential(rate=0.5),
            'distgamma0': tf.distributions.Gamma(concentration=1.0, rate=0.05),
            'distgamma1f': tf.distributions.Gamma(concentration=1.025, rate=0.025),
            'distgamma1': tf.distributions.Gamma(concentration=1.05, rate=0.05),
            'distgamma1s': tf.distributions.Gamma(concentration=1.1, rate=0.1),
            'distgamma2': tf.distributions.Gamma(concentration=1.1, rate=0.05),
        }
        pari2grad_dict = {
            'gradzero': lambda x: tf.ones_like(x) * tf.log(N+1) / N,
            'gradtail06': lambda x: 0.4 * tf.log(N+1) * tf.pow(R * x, -0.6) / (tf.pow(N+1, 0.4) - 1),
            'gradtail08': lambda x: 0.2 * tf.log(N+1) * tf.pow(R * x, -0.8) / (tf.pow(N+1, 0.2) - 1),
            'gradeven': lambda x: 1 / x,
            'gradhead12': lambda x: -0.2 * tf.log(N+1) * tf.pow(R * x, -1.2) / (tf.pow(N+1, -0.2) - 1),
            'gradhead14': lambda x: -0.4 * tf.log(N+1) * tf.pow(R * x, -1.4) / (tf.pow(N+1, -0.4) - 1),
            'gradhead16': lambda x: -0.6 * tf.log(N+1) * tf.pow(R * x, -1.6) / (tf.pow(N+1, -0.6) - 1),
            'gradhead18': lambda x: -0.8 * tf.log(N+1) * tf.pow(R * x, -1.8) / (tf.pow(N+1, -0.8) - 1),
            'gradhead20': lambda x: -1.0 * tf.log(N+1) * tf.pow(R * x, -2.0) / (tf.pow(N+1, -1.0) - 1),
        }
        score2loss_dict = {
            'zero': lambda x: x * tf.log(N+1) / N,
            'tail06': lambda x: (tf.pow(x * R, 0.4) - 1) * tf.log(N+1) / (tf.pow(N+1, 0.4) - 1),
            'tail08': lambda x: (tf.pow(x * R, 0.2) - 1) * tf.log(N+1) / (tf.pow(N+1, 0.2) - 1),
            'even': lambda x: tf.log(x),
            'head12': lambda x: (tf.pow(x * R, -0.2) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.2) - 1),
            'head14': lambda x: (tf.pow(x * R, -0.4) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.4) - 1),
            'head16': lambda x: (tf.pow(x * R, -0.6) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.6) - 1),
            'head18': lambda x: (tf.pow(x * R, -0.8) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.8) - 1),
            'head20': lambda x: (tf.pow(x * R, -1.0) - 1) * tf.log(N+1) / (tf.pow(N+1, -1.0) - 1),
        }
        if self.weight_type.startswith('pdf'):
            print('----- use pdf dist weight')
            dist = distribution_dict[self.weight_type[3:]]
            # rk_loss = tf.reduce_mean(tf.stop_gradient(dist.prob(hard_rkpct_i)) * rkpct_i)
            rk_loss = tf.reduce_mean(tf.stop_gradient(dist.prob(hard_rkpct_i)) * rk_i)
        elif self.weight_type.startswith('cdf'):
            print('----- use cdf dist weight')
            dist = distribution_dict[self.weight_type[3:]]
            rk_loss = tf.reduce_mean(dist.cdf(rkpct_i))
        elif self.weight_type.startswith('grad'):
            # print('!!!!!!!! ----- hard weight and splus rki')
            # rk_loss = tf.reduce_mean(tf.stop_gradient(pari2grad_dict[self.weight_type](hard_rk_i)) * rk_i)
            print('!!!!! ----- sigmoid weight and splus rki !!!!!')
            sig_rk_i = tf.reduce_sum(kernel_dict['sigmoid'](pair_diff * -self.boost_ratio), -1) + 1 # [b, ]
            rk_loss = tf.reduce_mean(tf.stop_gradient(pari2grad_dict[self.weight_type](sig_rk_i)) * rk_i)
        else:
            print("----- use original weight")
            rk_loss = tf.reduce_mean(score2loss_dict[self.weight_type](rk_i))

        # print('----- use test dist weight')
        # tempdist_dict = {
        #     'test1': tf.distributions.Exponential(rate=0.00001),
        #     'test2': tf.distributions.Exponential(rate=0.00002),
        #     'test3': tf.distributions.Exponential(rate=0.00003),
        #     'test5': tf.distributions.Exponential(rate=0.00005),
        # }
        # dist = tempdist_dict[self.weight_type]
        # rk_loss = tf.reduce_mean(tf.stop_gradient(dist.prob(hard_rkpct_i * R)) * R * rk_i)

        self.loss = rk_loss
        if self.l2_reg > 0:
            l2_loss = 0.0
            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
            for var in tf.trainable_variables():
                l2_loss += regularizer(var)
            self.loss += l2_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.mean_pair_diff = tf.reduce_mean(pair_diff)
        self.mean_rkpct = tf.reduce_mean(rk_i) / (M / 100)
        self.mean_hard_rkpct = tf.reduce_mean(hard_rk_i) / (M / 100)

    def check_gradient(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3]
        }
        item_emb_grd, urb_emb_grd = tf.gradients(self.loss, [self.item_eb, self.item_his_eb]) # debug: rebuild graph
        ied, ued = sess.run([item_emb_grd, urb_emb_grd], feed_dict=feed_dict)
        return np.mean(np.abs(ied), 0), np.mean(np.mean(np.abs(ued), 0), 0)

    def check_rkpct(self, item_emb, user_emb):
        # check rank percent (top x%)
        # neg_samples = np.random.randint(0, self.n_mid, 10000)
        # neg_samples = tf.convert_to_tensor(neg_samples)
        neg_samples = self.neg_sample_ph

        neg_embs = tf.nn.embedding_lookup(self.mid_embeddings_var, neg_samples) # (neg_num, dim)
        if self.unit:
            neg_norm = tf.reduce_sum(tf.square(neg_embs), -1) + 1e-8
            neg_embs = neg_embs / tf.expand_dims(tf.sqrt(neg_norm), -1)

        pos_logit = tf.reduce_sum(user_emb * item_emb, -1, keepdims=True) # (b, 1)
        neg_logit = tf.matmul(user_emb, neg_embs, transpose_b=True) # (b, n)
        pair_diff = pos_logit - neg_logit # (b, n)
        hard_rk_i = tf.reduce_sum(tf.nn.relu(tf.sign(-pair_diff)), -1) + 1
        self.mean_rkpct_check = tf.reduce_mean(hard_rk_i) / ((self.neg_num * self.batch_size) / 100)

    def train_pt(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.lr: inps[4],
            self.neg_sample_ph: inps[5]
        }
        values = sess.run([self.loss, self.mean_rkpct, self.mean_hard_rkpct, self.optimizer], feed_dict=feed_dict)
        return values[0], values[1], values[2] # loss, rank_percent, hard_rank_percent

    def build_output(self):
        if self.unit:
            # item
            item_norm = tf.reduce_sum(tf.square(self.mid_embeddings_var), -1) + 1e-8
            self.mid_embs_vars = self.mid_embeddings_var / tf.expand_dims(tf.sqrt(item_norm), -1)
            # user
            if len(self.user_eb.get_shape()) == 2:
                user_norm = tf.reduce_sum(tf.square(self.user_eb), -1) + 1e-8
                self.output_user_eb = self.user_eb / tf.expand_dims(tf.sqrt(user_norm), -1)
            elif len(self.user_eb.get_shape()) == 3:
                ni = self.user_eb.shape[1]
                user_eb_temp = tf.reshape(self.user_eb, [-1, self.dim])
                user_norm = tf.reduce_sum(tf.square(user_eb_temp), -1) + 1e-8
                norm_user_eb = user_eb_temp / tf.expand_dims(tf.sqrt(user_norm), -1)
                self.output_user_eb = tf.reshape(norm_user_eb, [-1, ni, self.dim])
            else:
                assert False
        else:
            # item
            self.mid_embs_vars = self.mid_embeddings_var
            # user
            self.output_user_eb = self.user_eb

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embs_vars)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.output_user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1]
        })
        return user_embs

    def output_loss(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.neg_sample_ph: inps[4]
        }
        loss, mean_rkpct_check = sess.run([self.loss, self.mean_rkpct_check], feed_dict=feed_dict)
        return loss, mean_rkpct_check

    def output_pairloc(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.neg_sample_ph: inps[4]
        }
        values = sess.run([self.pair_located], feed_dict=feed_dict)
        return values[0]

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len=256, l2_reg=0,
                 boost_ratio=1, kernel_type='splus', weight_type='even', unit=True):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len, flag="DNN", l2_reg=l2_reg,
            boost_ratio=boost_ratio, kernel_type=kernel_type, weight_type=weight_type, unit=unit)

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
        self.build_sampled_pt_loss(self.item_eb, self.user_eb)
        self.build_output()

class Model_GRU4REC(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len=256, l2_reg=0,
                 boost_ratio=1, kernel_type='splus', weight_type='even', unit=True):
        super(Model_GRU4REC, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len, flag="GRU4REC", l2_reg=l2_reg,
            boost_ratio=boost_ratio, kernel_type=kernel_type, weight_type=weight_type, unit=unit)
        with tf.name_scope('rnn_1'):
            self.sequence_length = self.mask_length
            rnn_outputs, final_state1 = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")

        self.user_eb = final_state1
        self.build_sampled_pt_loss(self.item_eb, self.user_eb)
        self.build_output()


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout

class Model_MIND(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, neg_num, num_interest, seq_len=256, hard_readout=True, relu_layer=True, l2_reg=0,
                 boost_ratio=1, kernel_type='splus', weight_type='even', unit=True):
        super(Model_MIND, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len, flag="MIND", l2_reg=l2_reg,
            boost_ratio=boost_ratio, kernel_type=kernel_type, weight_type=weight_type, unit=unit)

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=0, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)

        self.build_sampled_pt_loss(self.item_eb, readout)
        self.build_output()

class Model_ComiRec_DR(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, neg_num, num_interest, seq_len=256, hard_readout=True, relu_layer=False, l2_reg=0,
                 boost_ratio=1, kernel_type='splus', weight_type='even', unit=True):
        super(Model_ComiRec_DR, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len, flag="ComiRec_DR", l2_reg=l2_reg,
            boost_ratio=boost_ratio, kernel_type=kernel_type, weight_type=weight_type, unit=unit)

        item_his_emb = self.item_his_eb

        capsule_network = CapsuleNetwork(hidden_size, seq_len, bilinear_type=2, num_interest=num_interest, hard_readout=hard_readout, relu_layer=relu_layer)
        self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)

        self.build_sampled_pt_loss(self.item_eb, readout)
        self.build_output()

class Model_ComiRec_SA(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, neg_num, num_interest, seq_len=256, add_pos=True, l2_reg=0,
                 boost_ratio=1, kernel_type='splus', weight_type='even', unit=True):
        super(Model_ComiRec_SA, self).__init__(n_mid, embedding_dim, hidden_size, batch_size, neg_num, seq_len, flag="ComiRec_SA", l2_reg=l2_reg,
            boost_ratio=boost_ratio, kernel_type=kernel_type, weight_type=weight_type, unit=unit)
        
        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = num_interest
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            interest_emb = tf.matmul(item_att_w, item_list_emb)

        self.user_eb = interest_emb

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))
        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads)
        
        # readout = tf.layers.dense(tf.reshape(interest_emb, [-1, num_interest * self.dim]), hidden_size, activation=None)
        # self.user_eb = readout
        self.build_sampled_pt_loss(self.item_eb, readout)
        self.build_output()


def get_model(dataset, model_type, item_count, batch_size, maxlen, args):
    if model_type == 'DNN':
        model = Model_DNN(item_count, args.embedding_dim, args.hidden_size, batch_size, args.neg_samples, maxlen, l2_reg=args.l2_reg,
                          boost_ratio=args.boost_ratio, kernel_type=args.kernel_type, weight_type=args.weight_type, unit=args.unit)
    elif model_type == 'GRU4REC':
        model = Model_GRU4REC(item_count, args.embedding_dim, args.hidden_size, batch_size, args.neg_samples, maxlen, l2_reg=args.l2_reg,
                              boost_ratio=args.boost_ratio, kernel_type=args.kernel_type, weight_type=args.weight_type, unit=args.unit)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = Model_MIND(item_count, args.embedding_dim, args.hidden_size, batch_size, args.neg_samples, args.num_interest, maxlen,
                           relu_layer=relu_layer, l2_reg=args.l2_reg,
                           boost_ratio=args.boost_ratio, kernel_type=args.kernel_type, weight_type=args.weight_type, unit=args.unit)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(item_count, args.embedding_dim, args.hidden_size, batch_size, args.neg_samples, args.num_interest, maxlen, l2_reg=args.l2_reg,
                                 boost_ratio=args.boost_ratio, kernel_type=args.kernel_type, weight_type=args.weight_type, unit=args.unit)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.neg_samples, args.num_interest, maxlen, l2_reg=args.l2_reg,
                                 boost_ratio=args.boost_ratio, kernel_type=args.kernel_type, weight_type=args.weight_type, unit=args.unit)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    return model
