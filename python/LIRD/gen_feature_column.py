from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.contrib.layers.python.layers.feature_column import _EmbeddingColumn


import collections
import math
import numpy as np

from tensorflow.contrib.layers.python.layers.feature_column import _FeatureColumn, _RealValuedColumn, \
    _LazyBuilderByColumnsToTensor, _DeepEmbeddingLookupArguments
from tensorflow.contrib.layers.python.ops import bucketization_op
from tensorflow.contrib.layers.python.ops import sparse_ops
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_py
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.random_ops import random_normal


feature_key = 'features'
value_type_key = 'value_type'
feature_type_key = 'feature_type'
feature_name_key = 'feature_name'

def gen_feature(feature_conf):
    name = feature_conf[feature_name_key]
    value_type = feature_conf[value_type_key]

    if "vocab_size" in feature_conf:
        id_feature = fc.sparse_column_with_keys(
            column_name=name,
            keys=range(feature_conf['vocab_size']),
            dtype=tf.string
        )

        return fc._EmbeddingColumn(
            id_feature,
            dimension=feature_conf['embedding_dimension'],
            shared_embedding_name=feature_conf.get(feature_name_key),
        )
    elif "hash_bucket_size" in feature_conf \
            and "embedding_dimension" not in feature_conf:
        if value_type == "Int":
            id_feature = layers.sparse_column_with_integerized_feature(
                column_name=name,
                bucket_size=feature_conf['hash_bucket_size'],
                combiner=_get_combiner(feature_conf),
                # use_hashmap=use_hashmap
            )
        else:
            id_feature = layers.sparse_column_with_hash_bucket(
                column_name=name,
                hash_bucket_size=feature_conf['hash_bucket_size'],
                combiner=_get_combiner(feature_conf),
                # use_hashmap=use_hashmap
            )
        return id_feature
    elif "embedding_dimension" in feature_conf \
            and "hash_bucket_size" in feature_conf \
            and "boundaries" not in feature_conf \
            and "vocabulary_file" not in feature_conf:
        if value_type == "Int":
            return _EmbeddingColumn(
                sparse_id_column=layers.sparse_column_with_integerized_feature(
                    column_name=name,
                    bucket_size=feature_conf['hash_bucket_size'],
                    combiner=_get_combiner(feature_conf),
                    # use_hashmap=use_hashmap
                ),
                dimension=feature_conf['embedding_dimension'],
                combiner=_get_combiner(feature_conf),
                shared_embedding_name=feature_conf.get('shared_name', None)
            )
        else:
            id_feature = layers.sparse_column_with_hash_bucket(
                column_name=name,
                hash_bucket_size=feature_conf['hash_bucket_size'],
                # use_hashmap=use_hashmap
            )
            return _EmbeddingColumn(
                id_feature,
                dimension=feature_conf['embedding_dimension'],
                combiner=_get_combiner(feature_conf),
                shared_embedding_name=feature_conf.get('shared_name', None),
                max_norm=None
            )
    elif "embedding_dimension" in feature_conf \
            and "boundaries" not in feature_conf and "vocabulary_file" in feature_conf:
        use_hashmap = feature_conf.get("use_hashmap", False)
        if value_type == "Int":
            raise Exception("embedding with vocabulary_file does not support Int type")
        else:
            id_feature = fc.sparse_column_with_vocabulary_file(
                column_name=name,
                vocabulary_file=feature_conf["vocabulary_file"],
                num_oov_buckets=feature_conf["num_oov_buckets"],
                vocab_size=feature_conf["vocab_size"],
            )
            return _EmbeddingColumn(
                id_feature,
                dimension=feature_conf['embedding_dimension'],
                combiner=_get_combiner(feature_conf),
                shared_embedding_name=feature_conf.get('shared_name', None),
                max_norm=None
            )
    elif "embedding_dimension" in feature_conf \
            and "boundaries" in feature_conf:
        return embedding_bucketized_column(
            layers.real_valued_column(
                column_name=name,
                dimension=feature_conf.get('dimension', 1),
                default_value=[0.0 for _ in range(int(feature_conf.get('dimension', 1)))]),
            boundaries=[float(b) for b in feature_conf['boundaries'].split(',')],
            embedding_dimension=feature_conf["embedding_dimension"],
            max_norm=None,
            shared_name=feature_conf.get('shared_name', None),
            add_random=feature_conf.get('add_random', False)
        )
    elif "embedding_dimension" not in feature_conf \
            and "boundaries" in feature_conf:
        return layers.bucketized_column(
            layers.real_valued_column(
                column_name=name,
                dimension=feature_conf.get('dimension', 1),
                default_value=[0.0 for _ in range(int(feature_conf.get('dimension', 1)))]
            ),
            boundaries=[float(b) for b in feature_conf['boundaries'].split(',')]
        )
    else:
        return layers.real_valued_column(
            column_name=name,
            dimension=feature_conf.get('dimension', 1),
            default_value=[0.0 for _ in range(int(feature_conf.get('dimension', 1)))],
            normalizer=None if 'l2_norm' not in feature_conf else lambda x: tf.nn.l2_normalize(x, dim=-1)

        )


def _get_combiner(feature_conf):
    if "combiner" in feature_conf:
        return feature_conf["combiner"]
    else:
        return "mean"



class _EmbeddingBucketizedColumn(
    _FeatureColumn,
    fc_core._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple("_EmbeddingBucketizedColumn", ["source_column",
                                                          "boundaries", "embedding_dimension", "shared_name",
                                                          "max_norm", "trainable", "initializer",
                                                          "add_random"])):
    """Represents a bucketization transformation also known as binning.

    Instances of this class are immutable. Values in `source_column` will be
    bucketized based on `boundaries`.
    For example, if the inputs are:
        boundaries = [0, 10, 100]
        source_column = [[-5], [150], [10], [0], [4], [19]]

    then the bucketized feature will be:
        output = [[0], [3], [2], [1], [1], [2]]

    Attributes:
      source_column: A _RealValuedColumn defining dense column.
      boundaries: A list or tuple of floats specifying the boundaries. It has to
        be sorted. [a, b, c] defines following buckets: (-inf., a), [a, b),
        [b, c), [c, inf.)
      embedding_dimension: An integer specifying dimension of the embedding.
      shared_name: (Optional). A string specifying the name of shared
        embedding weights. This will be needed if you want to reference the shared
        embedding separately from the generated `_EmbeddingBucketizedColumn`.
    Raises:
      ValueError: if 'boundaries' is empty or not sorted.
    """

    def __new__(cls, source_column, boundaries, embedding_dimension, shared_name, max_norm, trainable, initializer,
                add_random):
        if not isinstance(source_column, _RealValuedColumn):
            raise TypeError("source_column must be an instance of _RealValuedColumn. "
                            "source_column: {}".format(source_column))

        if source_column.dimension is None:
            raise ValueError("source_column must have a defined dimension. "
                             "source_column: {}".format(source_column))

        if (not isinstance(boundaries, list) and
            not isinstance(boundaries, tuple)) or not boundaries:
            raise ValueError("boundaries must be a non-empty list or tuple. "
                             "boundaries: {}".format(boundaries))

        # We allow bucket boundaries to be monotonically increasing
        # (ie a[i+1] >= a[i]). When two bucket boundaries are the same, we
        # de-duplicate.
        sanitized_boundaries = []
        for i in range(len(boundaries) - 1):
            if boundaries[i] == boundaries[i + 1]:
                continue
            elif boundaries[i] < boundaries[i + 1]:
                sanitized_boundaries.append(boundaries[i])
            else:
                raise ValueError("boundaries must be a sorted list. "
                                 "boundaries: {}".format(boundaries))
        sanitized_boundaries.append(boundaries[len(boundaries) - 1])
        return super(_EmbeddingBucketizedColumn, cls).__new__(cls, source_column,
                                                              tuple(sanitized_boundaries), embedding_dimension,
                                                              shared_name,
                                                              max_norm, trainable, initializer, add_random)

    @property
    def name(self):
        return "{}_embedding_bucketized".format(self.source_column.name)

    @property
    def length(self):
        """Returns total number of buckets."""
        return len(self.boundaries) + 1

    @property
    def config(self):
        return self.source_column.config

    @property
    def key(self):
        """Returns a string which will be used as a key when we do sorting."""
        return "{}".format(self)

    def _deep_embedding_lookup_arguments(self, input_tensor):
        if self.initializer is None:
            stddev = 1 / math.sqrt(self.length)
            initializer = init_ops.truncated_normal_initializer(
                mean=0.0, stddev=stddev)
        else:
            initializer = self.initializer
        return _DeepEmbeddingLookupArguments(
            input_tensor=self.to_sparse_tensor(input_tensor),
            weight_tensor=None,
            vocab_size=self.length,
            dimension=self.embedding_dimension,
            initializer=initializer,
            combiner=None,
            shared_embedding_name=self.shared_name,
            hash_key=None,
            max_norm=self.max_norm,
            trainable=self.trainable,
            origin_feature_tensor=None,
            bucket_size=self.length
        )

    def to_sparse_tensor(self, input_tensor):
        """Creates a SparseTensor from the bucketized Tensor."""
        dimension = self.source_column.dimension
        batch_size = array_ops.shape(input_tensor, name="shape")[0]

        if len(input_tensor.get_shape()) > 2:
            return sparse_ops.dense_to_sparse_tensor(
                input_tensor, ignore_value=-2 ** 31)

        if dimension > 1:
            i1 = array_ops.reshape(
                array_ops.tile(
                    array_ops.expand_dims(
                        math_ops.range(0, batch_size), 1, name="expand_dims"),
                    [1, dimension],
                    name="tile"), [-1],
                name="reshape")
            i2 = array_ops.tile(
                math_ops.range(0, dimension), [batch_size], name="tile")
            # Flatten the bucket indices and unique them across dimensions
            # E.g. 2nd dimension indices will range from k to 2*k-1 with k buckets
            bucket_indices = array_ops.reshape(
                input_tensor, [-1], name="reshape") + self.length * i2
        else:
            # Simpler indices when dimension=1
            i1 = math_ops.range(0, batch_size)
            i2 = array_ops.zeros([batch_size], dtype=dtypes.int32, name="zeros")
            bucket_indices = array_ops.reshape(input_tensor, [-1], name="reshape")

        indices = math_ops.to_int64(array_ops.transpose(array_ops.stack((i1, i2))))
        shape = math_ops.to_int64(array_ops.stack([batch_size, dimension]))
        sparse_id_values = sparse_tensor_py.SparseTensor(
            indices, bucket_indices, shape)

        return sparse_id_values

    # pylint: disable=unused-argument
    def _wide_embedding_lookup_arguments(self, input_tensor):
        raise ValueError("Column {} is not supported in linear models. "
                         "Please use sparse_column.".format(self))

    def _transform_feature(self, inputs):
        """Handles cross transformation."""
        # Bucketize the source column.
        if not self.add_random:
            return bucketization_op.bucketize(
                inputs.get(self.source_column),
                boundaries=list(self.boundaries),
                name="bucketize")
        else:
            rawts = inputs.get(self.source_column)
            tbn = np.asarray(self.boundaries[1:])
            if len(tbn) > 30:
                # noise =  min(np.median(tbn)-tbn[0],tbn[20:-20].std())/2.
                noise = tbn[10:-10].std() / 10.
                rndts = rawts + random_normal(array_ops.shape(rawts), 0, noise)

                return bucketization_op.bucketize(rndts,
                                                  boundaries=list(self.boundaries),
                                                  name="bucketize")
            else:
                return bucketization_op.bucketize(
                    rawts,
                    boundaries=list(self.boundaries),
                    name="bucketize")

    def insert_transformed_feature(self, columns_to_tensors):
        """Handles sparse column to id conversion."""
        columns_to_tensors[self] = self._transform_feature(
            _LazyBuilderByColumnsToTensor(columns_to_tensors))

    @property
    def _parse_example_spec(self):
        return self.config

    @property
    def _num_buckets(self):
        return self.length * self.source_column.dimension

    @property
    def _variable_shape(self):
        return tensor_shape.TensorShape(
            [self.length * self.source_column.dimension])

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        return self._to_dnn_input_layer(
            inputs.get(self), weight_collections, trainable)


def embedding_bucketized_column(source_column, boundaries, embedding_dimension, shared_name=None, max_norm=None,
                                trainable=True, initializer=None, add_random=False):
    """Creates a _EmbeddingBucketizedColumn for discretizing dense input.

    Args:
      source_column: A _RealValuedColumn defining dense column.
      boundaries: A list or tuple of floats specifying the boundaries. It has to
        be sorted.

    Returns:
      A _EmbeddingBucketizedColumn.

    Raises:
      ValueError: if 'boundaries' is empty or not sorted.
    """
    retv = _EmbeddingBucketizedColumn(source_column, boundaries, embedding_dimension, shared_name, max_norm=max_norm,
                                      trainable=trainable, initializer=initializer,
                                      add_random=add_random)
    return retv


def _is_variable(v):
    """Returns true if `v` is a variable."""
    return isinstance(v, (variables.Variable,
                          resource_variable_ops.ResourceVariable))


def _get_feature_config(feature_column):  # copy from tensorflow -- feature_column.py
    """Returns configuration for the base feature defined in feature_column."""
    if not isinstance(feature_column, _FeatureColumn):
        raise TypeError(
            "feature_columns should only contain instances of _FeatureColumn. "
            "Given column is {}".format(feature_column))
    return feature_column.config


def create_feature_spec_for_parsing(feature_columns):  # copy from tensorflow -- feature_column.py
    """Helper that prepares features config from input feature_columns.

    The returned feature config can be used as arg 'features' in tf.parse_example.

    Typical usage example:

    ```python
    # Define features and transformations
    feature_a = sparse_column_with_vocabulary_file(...)
    feature_b = real_valued_column(...)
    feature_c_bucketized = bucketized_column(real_valued_column("feature_c"), ...)
    feature_a_x_feature_c = crossed_column(
      columns=[feature_a, feature_c_bucketized], ...)

    feature_columns = set(
      [feature_b, feature_c_bucketized, feature_a_x_feature_c])
    batch_examples = tf.parse_example(
        serialized=serialized_examples,
        features=create_feature_spec_for_parsing(feature_columns))
    ```

    For the above example, create_feature_spec_for_parsing would return the dict:
    {
      "feature_a": parsing_ops.VarLenFeature(tf.string),
      "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
      "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
    }

    Args:
      feature_columns: An iterable containing all the feature columns. All items
        should be instances of classes derived from _FeatureColumn, unless
        feature_columns is a dict -- in which case, this should be true of all
        values in the dict.
    Returns:
      A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
    """
    if isinstance(feature_columns, dict):
        feature_columns = feature_columns.values()

    features_config = {}
    for column in feature_columns:
        features_config.update(_get_feature_config(column))
    return features_config


def make_place_holder_tensors_for_base_features(feature_columns):  # copy from tensorflow -- feature_column.py
    """Returns placeholder tensors for inference.

    Args:
      feature_columns: An iterable containing all the feature columns. All items
        should be instances of classes derived from _FeatureColumn.
    Returns:
      A dict mapping feature keys to SparseTensors (sparse columns) or
      placeholder Tensors (dense columns).
    """
    # Get dict mapping features to FixedLenFeature or VarLenFeature values.
    dict_for_parse_example = create_feature_spec_for_parsing(feature_columns)
    placeholders = {}
    for column_name, column_type in dict_for_parse_example.items():
        if isinstance(column_type, parsing_ops.VarLenFeature):
            # Sparse placeholder for sparse tensors.
            placeholders[column_name] = array_ops.sparse_placeholder(
                column_type.dtype, name="Placeholder_{}".format(column_name))
        else:
            # Simple placeholder for dense tensors.
            placeholders[column_name] = array_ops.placeholder(
                column_type.dtype,
                shape=(None, column_type.shape[0]),
                name="Placeholder_{}".format(column_name))
    return placeholders
