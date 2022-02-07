# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import layers
import attention
import json
from tensorflow.contrib.layers.python.layers import feature_column as fc
import os
from tensorflow.contrib import layers
import bisect
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)
default_emb = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'


class DIN(object):
    def __init__(self,  num_epochs=30, batch_size=128, use_din=True):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.use_din = use_din
        self.hidden_units = [256, 128, 64]
        self.emb_size = 128
        self.seq_len = 5

        self.data_cols = ['user', 'item', 'seq', 'count', 'userid', 'itemid', 'rating', 'label']

        self.data_dim = 0
        for col in self.data_cols:
            if col in ['user', 'item']:
                self.data_dim += self.emb_size
            elif col == 'seq':
                self.data_dim += self.seq_len*self.emb_size
            else:
                self.data_dim += 1
        print('data dim: %i' % self.data_dim)

        return

    def build_model(self, data_info = None):

        self.model = tf.estimator.Estimator(
            model_fn=my_model,
            params={
                'max_len': self.seq_len,
                'hidden_units': self.hidden_units,
                'use_din': self.use_din,
                'emb_size': self.emb_size
            }
        )

    def build_input(self, data_file, delim = ','):
        assert tf.gfile.Exists(data_file)

        def parse_csv(value):
            parsed_line = tf.decode_csv(value, record_defaults=[[0.0]]*self.data_dim, field_delim=delim)
            label = parsed_line[-1]

            del parsed_line[-1]
            features = parsed_line
            # for i in range(len(self.columns_default)):
            #     if self.columns_default[i] == '':
            #         features[i] = process_list_column(features[i])

            d = dict(zip([str(i) for i in range(self.data_dim-1)], features))
            return d, label

        dataset = tf.data.TextLineDataset(data_file)

        dataset = dataset.shuffle(buffer_size=self.batch_size*10)
        dataset = dataset.map(parse_csv)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.batch(self.batch_size)

        return dataset

    # def build_input(self, data_file, delim = ','):
    #     assert tf.gfile.Exists(data_file)
    #     features = {}
    #     def parse_csv(value):
    #         parsed_line = tf.decode_csv(value, record_defaults= [[0.0]]*898, field_delim = delim)
    #         label = parsed_line[-1]
    #
    #         del parsed_line[-1]
    #
    #         index = 0
    #         features["user"] = (tf.concat(tf.reshape(parsed_line[index:index+self.emb_size], [-1,1]), -1))
    #         index += self.emb_size
    #         features["item"] = (tf.concat(tf.reshape(parsed_line[index:index+self.emb_size],[-1,1]), -1))
    #         index += self.emb_size
    #         seq_emb = []
    #         for i in range(self.seq_len):
    #             seq_emb.append(
    #                 tf.concat(tf.reshape(parsed_line[index:index + self.emb_size], [-1, 1 ,1]), -1)
    #             )
    #
    #             index += self.emb_size
    #
    #         features["seq"] = tf.concat(seq_emb, 1)
    #         features['seq_len'] = parsed_line[index]
    #         return features, label
    #
    #     dataset = tf.data.TextLineDataset(data_file)
    #
    #     dataset = dataset.shuffle(buffer_size=self.batch_size*10)
    #     dataset = dataset.map(parse_csv)
    #     dataset = dataset.repeat(self.num_epochs)
    #     dataset = dataset.batch(self.batch_size)
    #
    #     return dataset


    def gen_data(self, path):
        train_file = os.path.join(path, "ml-100k/u.data")
        user_emb_file = os.path.join(path, "emb_data/user_emb.txt")
        item_emb_file = os.path.join(path, "emb_data/item_emb.txt")
        user_emb = {}
        item_emb = {}
        data = {}

        with open(user_emb_file, 'r') as user_f:
            for line in user_f.readlines():
                line = line.strip('\n')
                sp = line.split(":")
                user_emb[int(sp[0])] = sp[1]
            user_f.close()

        with open(item_emb_file, 'r') as item_f:
            for line in item_f.readlines():
                line = line.strip('\n')
                sp = line.split(":")
                item_emb[int(sp[0])] = sp[1]
            item_f.close()

        with open(train_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                sp = line.split("\t")
                key = int(sp[0])
                if key in data:
                    data[key].append( [int(sp[1]), int(sp[2]), int(sp[3])])
                else:
                    data[key] = [[int(sp[1]), int(sp[2]), int(sp[3])]]
            f.close()

        train_out = open(os.path.join(path,"train.txt"), 'w')
        test_out = open(os.path.join(path,"test.txt"), 'w')
        for key in data.keys():
            # key: userid, value: [itemid, rating, timestamp]
            value = data[key]
            value.sort(key=lambda x: x[2])
            #print("{} has {} items".format(key, len(value)))
            for i in range(len(value)):
                d = []

                count = 0

                for col in self.data_cols:
                    if col == 'user':
                        d.append(user_emb[key])
                    elif col == 'item':
                        d.append(item_emb[value[i][0]])
                    elif col == 'seq':

                        for j in range(i, 0, -1):
                            # build click sequence
                            if value[j][1] > 3:
                                d.append(item_emb[value[j][0]])
                                count += 1
                                if count == self.seq_len:
                                    break
                        if count < self.seq_len:
                            for j in range(count, self.seq_len):
                                d.append(default_emb)
                    elif col == 'count':
                        d.append(str(count))
                    elif col == 'userid':
                        d.append(str(key))
                    elif col == 'itemid':
                        d.append(str(value[i][0]))
                    elif col == 'rating':
                        d.append(str(value[i][1]))
                    elif col == 'label':
                        d.append('1' if value[i][1] > 3 else '0')
                    else:
                        pass

                s = ",".join(d)
                assert len(s.split(',')) == self.data_dim

                if i >= len(value) - 3:
                    test_out.write(s + "\n")
                else:
                    train_out.write(s + "\n")

        train_out.close()
        test_out.close()

        return

    def train(self, data_path, delim = ','):
        # a file name or a list of file name [path, label_data, user_data, item_data]
        return self.model.train(input_fn=lambda: self.build_input(data_path, delim))

    def eval(self,  data_path, delim = ','):
        return self.model.evaluate(input_fn=lambda: self.build_input(data_path, delim))

    def predict(self, data_path, delim = ','):
        return self.model.predict(input_fn=lambda: self.build_input(data_path, delim))

def my_model(features, labels, mode, params):
    # user_feats = features['user']
    # item_feats = features['item']
    #
    # seq_feats = features['seq']
    # seq_len = features['seq_len']
    emb_size = params['emb_size']
    seq_len = params['max_len']
    index = 0
    user_feats = tf.concat([tf.reshape(features[str(i)], [-1, 1]) for i in range(index, index + emb_size)], -1)
    # user_feats = tf.reshape(user_feats, [-1,emb_size])
    index = index + emb_size
    item_feats = tf.concat([tf.reshape(features[str(i)], [-1, 1]) for i in range(index, index + emb_size)], -1)
    # item_feats = tf.reshape(item_feats, [-1,emb_size])
    index = index + emb_size

    seq_emb = []
    for i in range(seq_len):
        seq_emb.append(
            tf.concat([tf.reshape(features[str(i)], [-1, 1, 1]) for i in range(index, index + emb_size)], -1)
        )

        index += emb_size

    seq_feats = tf.concat(seq_emb, 1)
    seq_len = features[str(index)]
    index += 1

    userid = features[str(index)]
    index += 1
    itemid = features[str(index)]
    index += 1
    rating = features[str(index)]
    index += 1

    atten = attention.transformer_target_attention_layer(seq_feats, seq_len, item_feats, params['max_len'])

    if params['use_din']:
        net = tf.concat([user_feats, item_feats, atten],-1)
    else:
        net = tf.concat([user_feats, item_feats], -1)

    with tf.variable_scope("dnn"):
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)


    logits = layers.linear(net, 1, biases_initializer=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,
            'userid': userid,
            'itemid': itemid,
            'rating': rating
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1,1])

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels))

    auc = tf.metrics.auc(labels=labels, predictions=tf.nn.sigmoid(logits), name='auc')

    logging_hook = tf.train.LoggingTensorHook({"loss": loss, "auc": auc[1]}, every_n_iter=100)

    metrics = {'auc': auc}
    tf.summary.scalar('accuracy', auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


