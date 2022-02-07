import tensorflow as tf
from tensorflow.contrib import layers


def transformer_target_attention_layer(sequence,sequence_length,atten_query, max_len = 5):
  #         atten_query: [N, D]


    with tf.variable_scope("attention") as scope:
        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)  # [N, seq_length]

        sequence, stt_vec = multihead_attention(queries=sequence,
                                                keys=sequence,
                                                num_units=128,
                                                num_output_units=128,
                                                activation_fn=None,
                                                scope="self_attention",
                                                reuse=tf.AUTO_REUSE,
                                                query_masks=sequence_mask,
                                                key_masks=sequence_mask
                                                )

        atten_query = tf.expand_dims(atten_query, 1)  # [N, 1, D]
        atten_key = sequence
        atten_value = sequence

        # ua_item_vec: [N, 1, 128]
        item_vec, att_vec = multihead_target_attention(queries=atten_query,
                                                       keys=atten_key,
                                                       values=atten_value,
                                                       num_units=128,
                                                       num_output_units=128,
                                                       activation_fn=None,
                                                       scope="target_attention",
                                                       reuse=tf.AUTO_REUSE,
                                                       key_masks=sequence_mask
                                                       )

        dec = tf.reshape(item_vec, [-1, 128])  # [N, 128]
        return dec









def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=4,
                        scope="multihead_attention",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        atten_mode='base',
                        linear_projection=True,
                        variables_collections=None,
                        outputs_collections=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      num_output_units: A scalar. Output Value size.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
      query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
      key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys
      key_projection: A boolean, use projection to keys

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        query_len = queries.get_shape().as_list()[1]  # T_q
        key_len = keys.get_shape().as_list()[1]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[1], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (1, T_k, C)
            K = tf.reshape(K, [-1, keys.get_shape().as_list()[1], K.get_shape().as_list()[-1]])
            V = layers.fully_connected(keys_2d,
                                       num_output_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (1, T_k, C)
            V = tf.reshape(V, [-1, keys.get_shape().as_list()[1], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = keys

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)

        if atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        outputs = tf.where(key_masks, outputs, paddings)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.tile(tf.reshape(query_masks, [-1, query_len]), [num_heads, 1])  # (h*N, T_q)
        outputs = tf.reshape(outputs, [-1, key_len])  # (h*N*T_q, T_k)
        paddings = tf.zeros_like(outputs, dtype=tf.float32)  # (h*N*T_q, T_k)
        outputs = tf.where(tf.reshape(query_masks, [-1]), outputs,
                           paddings)  # tf.where((h*N*T_q), (h*N*T_q, T_k), (h*N*T_q, T_k)) => (h*N*T_q, T_k)
        outputs = tf.reshape(outputs, [-1, query_len, key_len])  # (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    return outputs, att_vec


def multihead_target_attention(queries,
                               keys,
                               values,
                               num_units=None,
                               num_output_units=None,
                               activation_fn=None,
                               num_heads=8,
                               scope="multihead_attention",
                               reuse=None,
                               key_masks=None,
                               atten_mode='base',
                               linear_projection=True,
                               variables_collections=None,
                               outputs_collections=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      num_output_units: A scalar. Output Value size.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
      query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
      key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys
      key_projection: A boolean, use projection to keys

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        if atten_mode == 'ln':
            activation_fn = None

        query_len = queries.get_shape().as_list()[1]  # T_q
        key_len = keys.get_shape().as_list()[1]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            values_2d = tf.reshape(values, [-1, values.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[1], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (1, T_k, C)
            K = tf.reshape(K, [-1, keys.get_shape().as_list()[1], K.get_shape().as_list()[-1]])
            V = layers.fully_connected(values_2d,
                                       num_output_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (1, T_k, C)
            V = tf.reshape(V, [-1, values.get_shape().as_list()[1], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = values

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)

        # Multiplication & Scale
        if atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        outputs = tf.where(key_masks, outputs, paddings)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    return outputs, att_vec

def feedforward(inputs,
                num_units=[2048, 512],
                activation_fn=None,
                scope="feedforward",
                reuse=None,
                variables_collections=None,
                outputs_collections=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = layers.fully_connected(inputs,
                                         num_units[0],
                                         activation_fn=activation_fn,
                                         variables_collections=variables_collections,
                                         outputs_collections=outputs_collections)
        outputs = layers.fully_connected(outputs,
                                         num_units[1],
                                         activation_fn=None,
                                         variables_collections=variables_collections,
                                         outputs_collections=outputs_collections)
        outputs += inputs

    return outputs
