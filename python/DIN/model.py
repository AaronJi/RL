import tensorflow as tf
import json
import os
from tensorflow.contrib import layers
import attention
# -*- coding: utf-8 -*-

import tensorflow as tf
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

    def build_model(self, data_info = None):

        self.model = tf.estimator.Estimator(
            model_fn=my_model,
            params={
                'max_len': self.seq_len,
                'hidden_units': self.hidden_units,
                'use_din':self.use_din
            }
        )

    def build_input(self, data_file, delim = ','):
        assert tf.gfile.Exists(data_file)

        def parse_csv(value):
            parsed_line = tf.decode_csv(value, record_defaults=[[0.0]]*898, field_delim=delim)
            label = parsed_line[-1]

            del parsed_line[-1]
            features = parsed_line
            # for i in range(len(self.columns_default)):
            #     if self.columns_default[i] == '':
            #         features[i] = process_list_column(features[i])

            d = dict(zip([str(i) for i in range(897)], features))
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
            value = data[key]
            value.sort(key = lambda x:x[2])
            print("{} has {} items".format(key, len(value)))
            for i in range(len(value)):

                d = []
                d.append(user_emb[key])
                d.append(item_emb[value[i][0]])
                count = 0
                for j in range(i, 0 , -1):
                    if value[j][1] > 3:
                        d.append(item_emb[value[j][0]])
                        count+=1
                        if count == self.seq_len:
                            break


                if count < self.seq_len:
                    for j in range(count, self.seq_len):
                        d.append(default_emb)
                d.append(str(count))
                d.append('1' if value[i][1] > 3 else '0')

                s = ",".join(d)
                if i >= len(value) - 3:
                    test_out.write(s +"\n")
                else:
                    train_out.write(s + "\n")

        train_out.close()
        test_out.close()




    def train(self, path, delim = ','):
        # a file name or a list of file name [path, label_data, user_data, item_data]
        return self.model.train(input_fn=lambda: self.build_input(os.path.join(path, 'train.txt'), delim))


    def eval(self,path, delim = ','):
        return self.model.evaluate(input_fn=lambda: self.build_input(os.path.join(path, 'test.txt'), delim))


    def predict(self,path, delim = ','):
        return self.model.predict(input_fn=lambda: self.build_input(os.path.join(path, 'test.txt'), delim))



def my_model(features, labels, mode, params):
    # user_feats = features['user']
    # item_feats = features['item']
    #
    # seq_feats = features['seq']
    # seq_len = features['seq_len']
    emb_size = 128
    seq_len = 5
    index = 0
    user_feats = tf.concat([tf.reshape(features[str(i)], [-1,1]) for i in range(index, index + emb_size)], -1)
    # user_feats = tf.reshape(user_feats, [-1,emb_size])
    index = index + emb_size
    item_feats = tf.concat([tf.reshape(features[str(i)],[-1,1]) for i in range(index, index + emb_size)], -1)
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


    max_len = params['max_len']


    atten = attention.transformer_target_attention_layer(seq_feats, seq_len, item_feats,max_len)

    if params['use_din']:
        net = tf.concat([user_feats, item_feats, atten],-1)
    else:
        net = tf.concat([user_feats, item_feats], -1)

    with tf.variable_scope("dnn"):
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)


    logits = layers.linear(
                    net,
                    1,
                    biases_initializer=None
                )


    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    labels = tf.reshape(tf.cast(labels, tf.float32), [-1,1])

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels))




    auc = tf.metrics.auc(labels=labels,
                                   predictions=tf.nn.sigmoid(logits),
                                   name='auc')

    logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                               "auc": auc[1]}, every_n_iter=100)

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


