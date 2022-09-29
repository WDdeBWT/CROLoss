#coding:utf-8
print('CODE-VERSION:', 1100)
import os
import sys
import csv
import time
import math
import pickle
import shutil
import random
import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf
has_gpu = tf.test.gpu_device_name() != ''

from model import get_model
from data_iterator import DataIterator

# SEED = 714
# tf.set_random_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao | almm')
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=256) # 256 for book, 512 for taobao
parser.add_argument('--neg_samples', type=int, default=10)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='DNN', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.02, help='') # 0.02 for rkpct_i, 0.001 for rk_i
parser.add_argument('--half_setp', type=int, default=1000, help='(k)')
parser.add_argument('--test_iter', type=int, default=4000, help='(k)')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--boost_ratio', type=float, default=10)
parser.add_argument('--kernel_type', type=str, default='sigmoid')
parser.add_argument('--weight_type', type=str, default='even')
parser.add_argument('--loss_type', type=str, default='pt')
parser.add_argument('--unit', type=int, default=1, help='')
parser.add_argument('--l2_reg', type=float, default=0, help='')
parser.add_argument('--fast_test', type=int, default=0, help='')

def flatten_valid_samples(user_ids, item_ids, clk_flags, urb_bhs, urb_masks):
    clk_flags = clk_flags if clk_flags is not None else [None for i in range(len(item_ids))]
    new_user_ids, new_item_ids, new_clk_flags, new_urb_bhs, new_urb_masks = [], [], [], [], []
    for uid, iid_list, clk_list, urb, urb_mask in zip(user_ids, item_ids, clk_flags, urb_bhs, urb_masks):
        clk_list = clk_list if clk_list is not None else [None for i in range(len(iid_list))]
        for iid, clk in zip(iid_list, clk_list):
            new_user_ids.append(uid)
            new_item_ids.append(iid)
            new_clk_flags.append(clk)
            new_urb_bhs.append(urb.copy())
            new_urb_masks.append(urb_mask.copy())
    return new_user_ids, new_item_ids, new_clk_flags, new_urb_bhs, new_urb_masks

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    # user_id [b,], target_item_id [b,], urb_id [b,l], urb_mask [b,l]
    return nick_id, item_id, hist_item, hist_mask

def check_pairloc_dist(check_uids, neg_sample, pairlocs_mat, ground_truth, pair_dist, pos_pair_dist):
    for uid, pairlocs in zip(check_uids, pairlocs_mat):
        for neg_id, pairloc in zip(neg_sample, pairlocs):
            # assert -1 <= pairloc and pairloc <= 1, 'pairloc error: ' + str(pairloc)
            idx = int((pairloc + 1) * 5)
            pair_dist[idx] += 1
            if neg_id in ground_truth[uid]:
                pos_pair_dist[idx] += 1

def evaluate_full(sess, dataset, test_data, model, model_path, batch_size, item_cate_map=None, save=True):
    eva_start_time = time.time()
    def recall_single(u_embs, i_ids, top_n):
        part_recall = 0.0
        part_hit = 0
        part_ndcg = 0.0
        D, I = faiss_index.search(u_embs, top_n)
        for i, iid_list in enumerate(i_ids):
            recall = 0
            dcg = 0.0
            true_item_set = set(iid_list)
            for no, iid in enumerate(I[i]):
                if iid in true_item_set:
                    recall += 1
                    dcg += 1.0 / math.log(no+2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)
            part_recall += recall * 1.0 / len(iid_list)
            if recall > 0:
                part_ndcg += dcg / idcg
                part_hit += 1
        return part_recall, part_hit, part_ndcg

    def recall_multi(u_embs, i_ids, top_n):
        part_recall = 0.0
        part_hit = 0
        part_ndcg = 0.0
        ni = u_embs.shape[1]
        u_embs = np.reshape(u_embs, [-1, u_embs.shape[-1]]) # (b * ni, dim)
        D, I = faiss_index.search(u_embs, top_n)
        for i, iid_list in enumerate(i_ids):
            recall = 0
            dcg = 0.0
            item_list_set = set()
            item_cor_list = []
            item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
            item_list.sort(key=lambda x:x[1], reverse=True)
            for j in range(len(item_list)):
                if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                    item_list_set.add(item_list[j][0])
                    item_cor_list.append(item_list[j][0])
                    if len(item_list_set) >= top_n:
                        break

            true_item_set = set(iid_list)
            for no, iid in enumerate(item_cor_list):
                if iid in true_item_set:
                    recall += 1
                    dcg += 1.0 / math.log(no+2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no+2, 2)
            part_recall += recall * 1.0 / len(iid_list)
            if recall > 0:
                part_ndcg += dcg / idcg
                part_hit += 1
        return part_recall, part_hit, part_ndcg

    import faiss
    item_embs = model.output_item(sess)
    try:
        if has_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
            faiss_index.add(item_embs)
        else:
            faiss_index = faiss.IndexFlatL2(args.embedding_dim)
            faiss_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    times = 0
    total_loss = 0.0
    total_recall_20 = 0.0
    total_recall_50 = 0.0
    total_recall_100 = 0.0
    total_recall_200 = 0.0
    total_recall_500 = 0.0
    total_hit_20 = 0.0
    total_hit_50 = 0.0
    total_hit_100 = 0.0
    total_hit_200 = 0.0
    total_hit_500 = 0.0
    total_ndcg_20 = 0.0
    total_ndcg_50 = 0.0
    total_ndcg_100 = 0.0
    total_ndcg_200 = 0.0
    total_ndcg_500 = 0.0
    # total_ieg = np.zeros(args.embedding_dim)
    # total_ueg = np.zeros(args.embedding_dim)
    total_mean_rkpct = 0.0

    for datas in test_data:
        nick_id, item_ids, hist_item, hist_mask, clk_flags = prepare_data(datas[0], datas[1]) + (None, )

        new_user_ids, new_item_ids, new_clk_flags, new_urb_bhs, new_urb_masks = flatten_valid_samples(nick_id, item_ids, clk_flags, hist_item, hist_mask)

        neg_sample = np.random.randint(0, item_count, args.neg_samples * batch_size)
        loss, mean_rkpct = model.output_loss(sess, [new_user_ids, new_item_ids, new_urb_bhs, new_urb_masks, neg_sample])
        total_loss += loss
        total_mean_rkpct += mean_rkpct

        # if times % 100 == 0:
        #     ieg, ueg = model.check_gradient(sess, [new_user_ids, new_item_ids, new_urb_bhs, new_urb_masks])
            # total_ieg += ieg
            # total_ueg += ueg

        user_embs = model.output_user(sess, [hist_item, hist_mask])
        if len(user_embs.shape) == 2:
            part_recall_20, part_hit_20, part_ndcg_20 = recall_single(user_embs, item_ids, 20)
            part_recall_50, part_hit_50, part_ndcg_50 = recall_single(user_embs, item_ids, 50)
            part_recall_100, part_hit_100, part_ndcg_100 = recall_single(user_embs, item_ids, 100)
            part_recall_200, part_hit_200, part_ndcg_200 = recall_single(user_embs, item_ids, 200)
            part_recall_500, part_hit_500, part_ndcg_500 = recall_single(user_embs, item_ids, 500)
        else:
            part_recall_20, part_hit_20, part_ndcg_20 = recall_multi(user_embs, item_ids, 20)
            part_recall_50, part_hit_50, part_ndcg_50 = recall_multi(user_embs, item_ids, 50)
            part_recall_100, part_hit_100, part_ndcg_100 = recall_multi(user_embs, item_ids, 100)
            part_recall_200, part_hit_200, part_ndcg_200 = recall_multi(user_embs, item_ids, 200)
            part_recall_500, part_hit_500, part_ndcg_500 = recall_multi(user_embs, item_ids, 500)

        total_recall_20 += part_recall_20
        total_recall_50 += part_recall_50
        total_recall_100 += part_recall_100
        total_recall_200 += part_recall_200
        total_recall_500 += part_recall_500
        total_hit_20 += part_hit_20
        total_hit_50 += part_hit_50
        total_hit_100 += part_hit_100
        total_hit_200 += part_hit_200
        total_hit_500 += part_hit_500
        total_ndcg_20 += part_ndcg_20
        total_ndcg_50 += part_ndcg_50
        total_ndcg_100 += part_ndcg_100
        total_ndcg_200 += part_ndcg_200
        total_ndcg_500 += part_ndcg_500
        total += len(item_ids)
        times += 1

    recall_20 = total_recall_20 / total
    recall_50 = total_recall_50 / total
    recall_100 = total_recall_100 / total
    recall_200 = total_recall_200 / total
    recall_500 = total_recall_500 / total
    hitrate_20 = total_hit_20 * 1.0 / total
    hitrate_50 = total_hit_50 * 1.0 / total
    hitrate_100 = total_hit_100 * 1.0 / total
    hitrate_200 = total_hit_200 * 1.0 / total
    hitrate_500 = total_hit_500 * 1.0 / total
    ndcg_20 = total_ndcg_20 / total
    ndcg_50 = total_ndcg_50 / total
    ndcg_100 = total_ndcg_100 / total
    ndcg_200 = total_ndcg_200 / total
    ndcg_500 = total_ndcg_500 / total
    loss = total_loss / times
    mean_rkpct = total_mean_rkpct / times
    # ieg = total_ieg / ((times - 1) // 100 + 1)
    # ueg = total_ueg / ((times - 1) // 100 + 1)
    # print('Avg item gradient:', ieg, ', urb gradient:', ueg, ', times:', times)

    # print('↓ Evaluate time: %.1f s' % (time.time() - eva_start_time))
    return {
                'loss': loss,
                'mean_hard_rkpct': mean_rkpct
            }, {
                'recall_20': recall_20,
                'hitrate_20': hitrate_20,
                'ndcg_20': ndcg_20
            }, {
                'recall_50': recall_50,
                'hitrate_50': hitrate_50,
                'ndcg_50': ndcg_50
            }, {
                'recall_100': recall_100,
                'hitrate_100': hitrate_100,
                'ndcg_100': ndcg_100
            }, {
                'recall_200': recall_200,
                'hitrate_200': hitrate_200,
                'ndcg_200': ndcg_200
            }, {
                'recall_500': recall_500,
                'hitrate_500': hitrate_500,
                'ndcg_500': ndcg_500
            }


def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        test_iter = 50,
        model_type = 'DNN',
        lr = 0.001,
        half_setp = 100,
        patience = 20,
        bh_log = None):

    def print_log(iter, train_loss, metrics_basic, metrics_20, metrics_50, metrics_100, metrics_200, metrics_500, time_use, trials):
        basic_log_str = 'Time from start: %.1f mins, iter: %d, train loss: %.4f, trials: %d' % (time_use / 60.0, iter, train_loss, trials)
        # basic_log_str += ', Valid: ' + ', '.join([key + ': %.4f' % value for key, value in metrics_basic.items()])
        basic_log_str += ', Valid: loss: %.4f, mean_hard_rkpct: %.2f%%' % (metrics_basic['loss'], metrics_basic['mean_hard_rkpct'])
        print(basic_log_str)
        print(' ' + ',  '.join([key + ': %.4f' % value for key, value in metrics_20.items()]))
        print(' ' + ',  '.join([key + ': %.4f' % value for key, value in metrics_50.items()]))
        print(', '.join([key + ': %.4f' % value for key, value in metrics_100.items()]))
        print(', '.join([key + ': %.4f' % value for key, value in metrics_200.items()]))
        print(', '.join([key + ': %.4f' % value for key, value in metrics_500.items()]))

    exp_name = 'none'
    best_model_path = "best_model/" + exp_name + '/'

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
        valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1, fast_test=bool(args.fast_test))

        model = get_model(dataset, model_type, item_count, batch_size, maxlen, args)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        iter = 0
        best_metric = 0
        local_half_setp = half_setp
        print('training begin')
        start_time = time.time()
        try:
            # m_basic, m_20, m_50, m_100, m_200 = evaluate_full(sess, dataset, valid_data, model, best_model_path, batch_size)
            # print_log(0, 0, m_basic, m_20, m_50, m_100, m_200, (time.time() - start_time))
            loss_sum = 0.0
            train_rk_sum = 0.0
            train_hard_rk_sum = 0.0
            trials = 0
            for datas in train_data:
                if dataset == 'almm' or dataset == 'msnews':
                    nick_id, item_ids, clk_flags, hist_item, hist_mask = datas
                    data_iter = [nick_id, item_ids, hist_item, hist_mask]
                else:
                    data_iter = prepare_data(datas[0], datas[1])

                # loss = model.train(sess, list(data_iter) + [lr])
                neg_sample = np.random.randint(0, item_count, args.neg_samples * batch_size)
                # loss = model.train_pt(sess, list(data_iter) + [lr, neg_sample])
                loss, train_rk, train_hard_rk = model.train_pt(sess, list(data_iter) + [lr, neg_sample])
                loss_sum += loss
                train_rk_sum += train_rk
                train_hard_rk_sum += train_hard_rk
                iter += 1

                if iter % test_iter == 0:
                    m_basic, m_20, m_50, m_100, m_200, m_500 = evaluate_full(sess, dataset, valid_data, model, best_model_path, batch_size)
                    recall_100 = m_100['recall_100']
                    if recall_100 > best_metric:
                        best_metric = recall_100
                        trials = 0

                        # train_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        # train_pos_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        # for times, datas in enumerate(train_data):
                        #     if times == 32:
                        #         break
                        #     assert dataset == 'book'
                        #     data_iter = prepare_data(datas[0], datas[1])
                        #     neg_sample = np.random.randint(0, item_count, args.neg_samples * batch_size)
                        #     pairlocs_mat = model.output_pairloc(sess, list(data_iter) + [neg_sample])
                        #     check_pairloc_dist(data_iter[0], neg_sample, pairlocs_mat, train_data.ground_truth, train_pair_dist, train_pos_pair_dist)
                        # train_pair_dist = list(map(lambda x: round(100 * x / (sum(train_pair_dist) + 1e-4), 2), train_pair_dist))
                        # train_pos_pair_dist = list(map(lambda x: round(100 * x / (sum(train_pos_pair_dist) + 1e-4), 2), train_pos_pair_dist))

                        # valid_data.temp_train()
                        # valid_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        # valid_pos_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        # for times, datas in enumerate(valid_data):
                        #     if times == 32:
                        #         break
                        #     assert dataset == 'book'
                        #     data_iter = prepare_data(datas[0], datas[1])
                        #     neg_sample = np.random.randint(0, item_count, args.neg_samples * batch_size)
                        #     pairlocs_mat = model.output_pairloc(sess, list(data_iter) + [neg_sample])
                        #     check_pairloc_dist(data_iter[0], neg_sample, pairlocs_mat, valid_data.ground_truth, valid_pair_dist, valid_pos_pair_dist)
                        # valid_pair_dist = list(map(lambda x: round(100 * x / (sum(valid_pair_dist) + 1e-4), 2), valid_pair_dist))
                        # valid_pos_pair_dist = list(map(lambda x: round(100 * x / (sum(valid_pos_pair_dist) + 1e-4), 2), valid_pos_pair_dist))
                        # valid_data.temp_train(False)

                    else:
                        trials += 1
                        train_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        train_pos_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        valid_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        valid_pos_pair_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    print_log(iter, (loss_sum / test_iter), m_basic, m_20, m_50, m_100, m_200, m_500, (time.time() - start_time), trials)
                    # print(train_pair_dist, '%, train_pair_dist')
                    # print(train_pos_pair_dist, '%, train_pos_pair_dist')
                    # print(valid_pair_dist, '%, valid_pair_dist')
                    # print(valid_pos_pair_dist, '%, valid_pos_pair_dist')
                    print("↑ train_rkpct: %.2f%%, train_hard_rkpct: %.2f%%, total_neg_nums: %d" % (
                        train_rk_sum / test_iter, train_hard_rk_sum / test_iter, args.neg_samples * batch_size))
                    loss_sum = 0.0
                    train_rk_sum = 0.0
                    train_hard_rk_sum = 0.0

                    local_half_setp -= 1
                    if local_half_setp == 0:
                        local_half_setp = half_setp
                        test_iter = test_iter // 2
                        if test_iter <= 2000:
                            local_half_setp = 200
                        print(('----- current test iter:', test_iter))

                if trials >= patience:
                    print('-' * 89)
                    print('trials > patience')
                    break
                if iter >= 500000:
                    print('-' * 89)
                    print('iter >= 500000')
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # model.restore(sess, best_model_path)
        print('Current best recall_100', best_metric)
        # m_basic, m_20, m_50, m_100, m_200 = evaluate_full(sess, dataset, valid_data, model, best_model_path, batch_size, save=False)
        # print_log(0, 0, m_basic, m_20, m_50, m_100, m_200, (time.time() - start_time))

        # test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        # metrics = evaluate_full(sess, dataset, test_data, model, best_model_path, batch_size, save=False)
        # print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def test(
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = 'none'
    best_model_path = "best_model/" + exp_name + '/'
    # best_model_path = "../best_model_comirec/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    # item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, dataset, test_data, model, best_model_path, batch_size, save=False)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def output(
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = 'none'
    best_model_path = "best_model/" + exp_name + '/'
    # best_model_path = "../best_model_comirec/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('output/' + exp_name + '_emb.npy', item_embs)

if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    dataset = args.dataset
    batch_size = args.batch_size

    if dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        maxlen = 50
        bh_log = None
        train_file = path + dataset + '_train.txt'
        valid_file = path + dataset + '_valid.txt'
        test_file = path + dataset + '_test.txt'
        cate_file = path + dataset + '_item_cate.txt'
    elif dataset == 'book':
        path = './data/book_data/'
        item_count = 367983
        maxlen = 20
        bh_log = None
        train_file = path + dataset + '_train.txt'
        valid_file = path + dataset + '_valid.txt'
        test_file = path + dataset + '_test.txt'
        cate_file = path + dataset + '_item_cate.txt'

    print('item_count:', item_count)
    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file, 
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=args.test_iter, 
              model_type=args.model_type, lr=args.learning_rate, half_setp=args.half_setp, patience=args.patience, bh_log=bh_log)
    elif args.p == 'test':
        test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size, 
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, 
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')
