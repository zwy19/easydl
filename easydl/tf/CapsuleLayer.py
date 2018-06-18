import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os

def eye(shape):
    shape = list(shape)
    assert len(shape) >= 2
    eye = tf.eye(shape[-2], shape[-1])
    for _ in shape[:-2]:
        eye = tf.expand_dims(eye, axis=0)
    eye = tf.tile(eye, shape[:-2] + [1, 1])
    return eye

hinton_squash = (lambda x : (x ** 2) / (1 + x ** 2))

def general_squash(input_tensor, axis=-1, f=hinton_squash):
    norm = tf.sqrt(tf.reduce_sum(input_tensor ** 2, axis=axis, keep_dims=True))
    return (input_tensor / (norm + 1e-8)) * f(norm)

def _leaky_softmax(logits):
    """Adds extra dimmension to routing logits.

    This enables active capsules to be routed to the extra dim if they are not a
    good fit for any of the capsules in layer above.

    Args:
    logits: The original logits. shape is
      [batch_size, input_capsule_num, output_capsule_num]
    Returns:
    Routing probabilities for each pair of capsules. Same shape as logits.
    """

    output_dim = logits.shape.as_list()[2]

    # leak is a zero matrix with same shape as logits except dim(2) = 1 because of the reduce_sum.
    leak = tf.zeros_like(logits)
    leak = tf.reduce_sum(leak, axis=2, keep_dims=True)
    leaky_logits = tf.concat([leak, logits], axis=2)
    leaky_routing = tf.nn.softmax(leaky_logits, dim=2)
    return tf.split(leaky_routing, [1, output_dim], 2)[1]

def CapsuleLayer_tf(layer_in, dims_per_out_capsule=16, num_out_capsules=10, num_routing=3, use_leaky_softmax=True, eye_init=False, squash_func=tf.tanh):
    assert len(layer_in.shape.as_list()) == 3, 'layer_in must be 3-D tensor!'
    batchSize, num_in_capsules, dims_per_in_capsule= layer_in.shape.as_list()
    
    layer_in = tf.expand_dims(layer_in, -1)
    layer_in = tf.expand_dims(layer_in, 2)
    assert layer_in.shape.as_list() == [batchSize, num_in_capsules, 1, dims_per_in_capsule, 1], 'fatal error! dims not correct!'
    
    if eye_init:
        weight = tf.get_variable(name='weight',
                dtype=tf.float32, 
                initializer=eye((1, num_in_capsules, num_out_capsules, dims_per_out_capsule, dims_per_in_capsule)))
    else:
        weight = tf.get_variable(name='weight',
                        shape=(1, num_in_capsules, num_out_capsules, dims_per_out_capsule, dims_per_in_capsule), 
                        dtype=tf.float32, 
                        initializer=tf.truncated_normal_initializer(stddev=0.2))

    
    layer_in = tf.transpose(layer_in, [0, 1, 2, 4, 3])
    Uij = tf.reduce_sum(layer_in * weight, axis=-1, keep_dims=True)
    assert Uij.shape.as_list() == [batchSize, num_in_capsules, num_out_capsules, dims_per_out_capsule, 1], 'fatal error! dims not correct!'

    bij = tf.zeros_like(tf.reduce_sum(Uij, axis=3, keep_dims=True)) # to avoid explicitly use None as dimension
    for _ in range(num_routing):
        cij = _leaky_softmax(bij) if use_leaky_softmax else tf.nn.softmax(bij, dim=2)
        assert cij.shape.as_list() == [batchSize, num_in_capsules, num_out_capsules, 1, 1], 'fatal error! dims not correct!'

        Sj = tf.reduce_sum(cij * Uij, axis=1)
        assert Sj.shape.as_list() == [batchSize, num_out_capsules, dims_per_out_capsule, 1], 'fatal error! dims not correct!'

        Vj = general_squash(Sj, axis=2, f=squash_func)
        Vj = tf.expand_dims(Vj, axis=1)
        assert Vj.shape.as_list() == [batchSize, 1, num_out_capsules, dims_per_out_capsule, 1], 'fatal error! dims not correct!'

        bij += tf.reduce_sum(Uij * Vj, axis=3, keep_dims=True)
    
    Vj = tf.squeeze(Vj, axis=1)
    Vj = tf.squeeze(Vj, axis=-1)
    assert Vj.shape.as_list() == [batchSize, num_out_capsules, dims_per_out_capsule], 'fatal error! dims not correct!'
    return Vj

def CapsuleLayer_tl(layer_in, dims_per_out_capsule=16, num_out_capsules=10, num_routing=3,use_leaky_softmax=True, scope='capsule', eye_init=False, squash_func=tf.tanh):
    with tf.variable_scope(scope) as vs:
        layer_in = TileLayer(layer_in,multiples=[1 for _ in range(len(layer_in.outputs.shape.as_list()))], name=scope + '/capsule')
        layer_in.outputs = CapsuleLayer_tf(
            layer_in.outputs, 
            dims_per_out_capsule=dims_per_out_capsule, 
            num_out_capsules=num_out_capsules, 
            num_routing=num_routing,
            use_leaky_softmax=use_leaky_softmax,
            eye_init=eye_init,
            squash_func=squash_func,
        )
        layer_in.all_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)#for variable storing
        layer_in.all_layers[-1] = layer_in.outputs # then it can be caught in all_layers
        return layer_in