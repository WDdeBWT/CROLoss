# This is a simple code template to show how to deploy CROLoss in your retrieval model (to replace the softmax cross-entropy loss)

class TwoTowerModel(object):
    def __init__(self, user_features, context_features, pos_item_features, neg_item_features):
        self.boost_ratio = 10
        self.kernel_type = 'splus' # exp/splus/hinge/sigmoid
        self.weight_type = 'lambdaeven' # lambdatail08/lambdaeven/lambdahead12
        self.neg_num = 4
        self.batch_size = 4096
        self.lr = 0.001

        self.user_emb = self.tower_a(user_features, context_features)
        self.item_emb = self.tower_b(pos_item_features) # target item vector
        self.neg_embs = self.tower_b(neg_item_features) # negative sample item vector

    def tower_a(self, user_features, context_features):
        # User tower, output user representation vector
        pass

    def tower_b(self, item_features):
        # Item tower, output item representation vector
        pass

    def sotfmax_loss(self):
        # Conventional sampled softmax cross-entropy loss
        pass # e.g. tf.nn.sampled_softmax_loss

    def cro_loss(self):
        # Core code of CROLoss
        pos_logit = tf.reduce_sum(self.user_emb * self.item_emb, -1, keepdims=True) # (b, 1)
        neg_logit = tf.matmul(self.user_emb, self.neg_embs, transpose_b=True) # (b, n)
        pair_diff = pos_logit - neg_logit # (b, n)

        kernel_dict = {
            'exp': lambda x: tf.exp(x),
            'splus': lambda x: tf.log(1 + tf.exp(x)),
            'hinge': lambda x: tf.nn.relu(x + 0.5),
            'sigmoid': lambda x: tf.nn.sigmoid(x),
        }

        rk_i = tf.reduce_sum(kernel_dict[self.kernel_type](pair_diff * -self.boost_ratio), -1) + 1 # [b, ]

        N = tf.cast(1000000, dtype=tf.float32) # total num of item set
        M = self.neg_num * self.batch_size # total num of negative samples
        R = N / M

        pari2grad_dict = {
            'lambdazero': lambda x: tf.ones_like(x) * tf.log(N+1) / N,
            'lambdatail08': lambda x: 0.2 * tf.log(N+1) * tf.pow(R * x, -0.8) / (tf.pow(N+1, 0.2) - 1),
            'lambdaeven': lambda x: 1 / x,
            'lambdahead12': lambda x: -0.2 * tf.log(N+1) * tf.pow(R * x, -1.2) / (tf.pow(N+1, -0.2) - 1),
            'lambdahead14': lambda x: -0.4 * tf.log(N+1) * tf.pow(R * x, -1.4) / (tf.pow(N+1, -0.4) - 1),
            'lambdahead16': lambda x: -0.6 * tf.log(N+1) * tf.pow(R * x, -1.6) / (tf.pow(N+1, -0.6) - 1),
            'lambdahead18': lambda x: -0.8 * tf.log(N+1) * tf.pow(R * x, -1.8) / (tf.pow(N+1, -0.8) - 1),
            'lambdahead20': lambda x: -1.0 * tf.log(N+1) * tf.pow(R * x, -2.0) / (tf.pow(N+1, -1.0) - 1),
        }
        score2loss_dict = {
            'zero': lambda x: x * tf.log(N+1) / N,
            'tail08': lambda x: (tf.pow(x * R, 0.2) - 1) * tf.log(N+1) / (tf.pow(N+1, 0.2) - 1),
            'even': lambda x: tf.log(x),
            'head12': lambda x: (tf.pow(x * R, -0.2) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.2) - 1),
            'head14': lambda x: (tf.pow(x * R, -0.4) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.4) - 1),
            'head16': lambda x: (tf.pow(x * R, -0.6) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.6) - 1),
            'head18': lambda x: (tf.pow(x * R, -0.8) - 1) * tf.log(N+1) / (tf.pow(N+1, -0.8) - 1),
            'head20': lambda x: (tf.pow(x * R, -1.0) - 1) * tf.log(N+1) / (tf.pow(N+1, -1.0) - 1),
        }
        if self.weight_type.startswith('lambda'):
            print('----- lambda(sigmoid) weight')
            sig_rk_i = tf.reduce_sum(kernel_dict['sigmoid'](pair_diff * -self.boost_ratio), -1) + 1 # [b, ]
            rk_loss = tf.reduce_mean(tf.stop_gradient(pari2grad_dict[self.weight_type](sig_rk_i)) * rk_i)
        else:
            print("----- use original weight")
            rk_loss = tf.reduce_mean(score2loss_dict[self.weight_type](rk_i))

        self.loss = rk_loss

    def optimizer(self):
        return xxxOptimizer(learning_rate=self.lr).minimize(self.loss) # e.g. tf.train.AdamOptimizer
