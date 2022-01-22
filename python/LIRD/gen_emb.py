# -*- coding: utf-8 -*-

import tensorflow as tf
import json
from tensorflow.contrib.layers.python.layers import feature_column as fc
import os
import gen_feature_column
from tensorflow.contrib import layers
import bisect

tf.logging.set_verbosity(tf.logging.INFO)


class EmbeddingGeneration(object):
    def __init__(self, config_path, num_epochs = 100, batch_size=128):
        with open(config_path) as f:
            config = json.load(f)
            self.user_conf = config['user']
            self.item_conf = config['item']

        self.columns = []
        self.columns_default = []
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.user_feature_column = []
        self.item_feature_column = []
        self.user_columns = []
        self.item_columns = []
        self.feature_column = []
        self.user_hidden_units = [256, 128]
        self.item_hidden_units = [256, 128]
        self.emb_size = 128


    def build_model(self, data_info = None):
        self.build_feature_conf()
        if data_info is not None:
            [path,user_file, item_file, train_file, test_file] = data_info
            self.gen_data(path,user_file, item_file, train_file, test_file, "\t")

        self.model = tf.estimator.Estimator(
            model_fn=my_model,
            params={
                'user_columns': self.user_columns,
                'item_columns': self.item_columns,
                'user_hidden_units': self.user_hidden_units,
                'item_hidden_units': self.item_hidden_units,
                'user_feature_columns': self.user_feature_column,
                'item_feature_columns': self.item_feature_column,
                "emb_size": self.emb_size
            }
        )

    def build_input(self, data_file, delim = '\t'):
        assert tf.gfile.Exists(data_file)

        def parse_csv(value):
            parsed_line = tf.decode_csv(value, record_defaults= self.columns_default, field_delim = delim)
            label = parsed_line[-1]

            del parsed_line[-1]
            features = parsed_line
            # for i in range(len(self.columns_default)):
            #     if self.columns_default[i] == '':
            #         features[i] = process_list_column(features[i])

            d = dict(zip(self.columns, features))
            return d, label

        dataset = tf.data.TextLineDataset(data_file)

        dataset = dataset.shuffle(buffer_size=self.batch_size*10)
        dataset = dataset.map(parse_csv)
        dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.batch(self.batch_size)

        return dataset


    def gen_data(self, path,user_file, item_file, train_file, test_file, delim):
        train_data = []
        user_feat = []
        item_feat = []
        user_feat.append('')
        item_feat.append('')
        test_data = []
        train_out = os.path.join(path, 'train_wide.csv')
        test_out = os.path.join(path, 'test_wide.csv')

        user_predict = os.path.join(path, 'user_predict.csv')
        item_predict = os.path.join(path, 'item_predict.csv')

        with open(os.path.join(path, user_file), 'r') as user_f:
            for line in user_f.readlines():
                line = line.strip('\n')

                user_feat.append(self._process_user_feature(line))
            user_f.close()

        with open(os.path.join(path, item_file), 'r', encoding='ISO-8859-1') as item_f:
            for line in item_f.readlines():
                line = line.strip('\n')
                item_feat.append(self._process_item_feature(line))
            item_f.close()

        with open(os.path.join(path, train_file), "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                if len(line) > 0:
                    line_sp = line.split(delim)
                    label = int(line_sp[2])
                    label = 1 if label > 3 else 0
                    train_data.append(
                        delim.join([user_feat[int(line_sp[0])], item_feat[int(line_sp[1])], str(label)])
                    )

        with open(train_out, "w") as write_f:
            for d in train_data:
                write_f.write(d+"\n")


        with open(os.path.join(path, test_file), "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                if len(line) > 0:
                    line_sp = line.split(delim)
                    label = int(line_sp[2])
                    label = 1 if label > 3 else 0
                    test_data.append(
                        delim.join([user_feat[int(line_sp[0])], item_feat[int(line_sp[1])], str(label)])
                    )

        with open(test_out, "w") as write_f:
            for d in test_data:
                write_f.write(d + "\n")
            write_f.close()

        user_default_val = user_feat[1]
        item_default_val = item_feat[1]

        with open(user_predict, 'w') as f:
            for i in range(1, len(user_feat)):
                f.write(delim.join([user_feat[i], item_default_val, '0'])+"\n")


        with open(item_predict, 'w') as f:
            for i in range(1, len(item_feat)):
                f.write(delim.join([user_default_val, item_feat[i], '0'])+"\n")



    def _process_user_feature(self, line,delim = '\t'):
        [user_id,  age , gender,occupation,zip] = line.split("|")

        age = bisect.bisect_left([18,22,26,30,35,49,45,50,60],int(age))

        return delim.join([user_id,  str(age) , gender,occupation,zip])


    def _process_item_feature(self, line,delim = '\t'):
        [item_id, title, date, video_date,
         URL, unknown, Action, Adventure, Animation,
         Children , Comedy , Crime , Documentary , Drama , Fantasy ,
         Film_Noir, Horror, Musical, Mystery, Romance, Sci_Fi,
         Thriller, War, Western ] = line.split("|")

        date = date[-4:]

        return delim.join([item_id, date, unknown, Action, Adventure, Animation,
         Children , Comedy , Crime , Documentary , Drama , Fantasy ,
         Film_Noir, Horror, Musical, Mystery, Romance, Sci_Fi,
         Thriller, War, Western ])


    def train(self, path, delim = '\t'):
        # a file name or a list of file name [path, label_data, user_data, item_data]
        return self.model.train(input_fn=lambda: self.build_input(os.path.join(path, 'train_wide.csv'), delim))

    def predict(self,path,  delim = '\t'):

        with open(os.path.join(path, "user_emb.txt"), 'w') as writer:

            for data in self.model.predict(input_fn=lambda: self.build_input(os.path.join(path, 'user_predict.csv'), delim)):
                user_emb = data['user_emb']
                user_id = data['user_id']
                writer.write("{}:{}\n".format(self.get_str(user_id), self.get_str(user_emb)))

            writer.close()

        with open(os.path.join(path, "item_emb.txt"), 'w') as writer:

            for data in self.model.predict(
                    input_fn=lambda: self.build_input(os.path.join(path, 'item_predict.csv'), delim)):
                item_emb = data['item_emb']
                item_id = data['item_id']
                writer.write("{}:{}\n".format(self.get_str(item_id), self.get_str(item_emb)))

            writer.close()

    def get_str(self,input):
        return str(input)[2:-1]

    def eval(self,path, delim = '\t'):
        return self.model.evaluate(input_fn=lambda: self.build_input(os.path.join(path, 'test_wide.csv'), delim))


    def build_feature_conf(self):

        for feature_conf in self.user_conf:
            fc = self.gen_feature_column(feature_conf)
            self.user_feature_column.append(fc)
            self.feature_column.append(fc)
            self.user_columns.append(feature_conf['feature_name'])
            self.columns.append(feature_conf['feature_name'])
            self.columns_default.append(
                self.gen_def_val(feature_conf['value_type'])
            )

        for feature_conf in self.item_conf:
            fc = self.gen_feature_column(feature_conf)
            self.item_feature_column.append(fc)
            self.feature_column.append(fc)
            self.item_columns.append(feature_conf['feature_name'])
            self.columns.append(feature_conf['feature_name'])
            self.columns_default.append(
                self.gen_def_val(feature_conf['value_type'])
            )

        self.columns_default.append([0])




    def gen_def_val(self, value_type):
        if value_type == 'Int':
            return [0]
        elif value_type == 'Double':
            return [0.0]
        else:
            return ['']



    def gen_feature_column(self, feature_conf):
        feature_name = feature_conf['feature_name']

        if "comment" in feature_conf:
            return None

        if "vocab_size" in feature_conf:
            id_feature = fc.sparse_column_with_keys(
                column_name=feature_name,
                keys=[str(i) for i in range(feature_conf['vocab_size'])]

            )

            return fc._EmbeddingColumn(
                id_feature,
                dimension=feature_conf['embedding_dimension'],
                shared_embedding_name=feature_conf.get('name'),
            )
        elif 'hash_bucket_size' in feature_conf:
            id_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
                column_name=feature_name,
                hash_bucket_size=feature_conf['hash_bucket_size'],
                # use_hashmap=use_hashmap
            )
            return fc._EmbeddingColumn(
                id_feature,
                dimension=feature_conf['embedding_dimension'],
                shared_embedding_name=feature_conf.get('shared_name', None),
                max_norm=None)

        else:
            return tf.contrib.layers.real_valued_column(
                column_name=feature_name,
                dimension=feature_conf.get('dimension', 1),
                default_value=[0.0 for _ in range(int(feature_conf.get('dimension', 1)))],
                normalizer=None if 'l2_norm' not in feature_conf else lambda x: tf.nn.l2_normalize(x, dim=-1)

            )



def my_model(features, labels, mode, params):
    user_columns = params['user_columns']
    item_columns = params['item_columns']

    user_features = {}
    item_features = {}
    for name in user_columns:
        if name in features:
            user_features[name] = features[name]

    for name in item_columns:
        if name in features:
            item_features[name] = features[name]


    with tf.variable_scope("user_net") as scope1:


        user_net = tf.feature_column.input_layer(user_features, params['user_feature_columns'])
        for units in params['user_hidden_units']:
            user_net = tf.layers.dense(user_net, units=units, activation=tf.nn.relu)
        user_net = tf.layers.dense(user_net, params.get("emb_size"), activation=None)

        user_net_out = tf.reduce_join(tf.as_string(user_net, 8), -1, separator=',')


    with tf.variable_scope("item_net") as scope2:
        item_net = layers.input_from_feature_columns(item_features, params['item_feature_columns'])
        for units in params['item_hidden_units']:
            item_net = tf.layers.dense(item_net, units=units, activation=tf.nn.relu)

        item_net = tf.layers.dense(item_net, params.get("emb_size"), activation=None)
        item_net_out = tf.reduce_join(tf.as_string(item_net, 8), -1, separator=',')


    c = tf.layers.dense(tf.concat([user_net, item_net], -1), units=128, activation=tf.nn.relu)

    logits = layers.linear(
                    c,
                    1,
                    biases_initializer=None
                )


    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits,
            'user_emb':user_net_out,
            'item_emb':item_net_out,
            'user_id': features['user_id'],
            "item_id": features['item_id']
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


