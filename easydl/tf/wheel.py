import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

EPSILON = 1e-20

def get_initialized_session():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess

def TFBinaryCrossEntropy(predict, label, epsilon=EPSILON):
    assert label.shape == predict.shape
    axis = list(range(len(tf.shape(predict))))
    axis = axis[1:]
    loss = tf.reduce_mean(tf.reduce_sum(-label * tf.log(predict + epsilon) - (1.0 - label) * tf.log(1.0 - predict + epsilon), axis=axis))
    return loss

def MagnifyLayer(layer_in, scope='maynify'):
    with tf.variable_scope(scope) as vs:
        layer_in = TileLayer(layer_in,multiples=[1 for _ in range(len(layer_in.outputs.shape.as_list()))], name=scope + '/maynify')
        shape = layer_in.outputs.shape.as_list()
        shape[0] = 1
        weight = tf.get_variable(name='weight',
                        shape=shape, 
                        dtype=tf.float32, 
                        initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0))
        layer_in.outputs = (1.0 + weight ** 2) * layer_in.outputs
        layer_in.all_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)#for variable storing
        layer_in.all_layers[-1] = layer_in.outputs # then it can be caught in all_layers
        return layer_in
    
    
def IdentityLayer(layer_in, name):
    return TileLayer(layer_in,multiples=[1 for _ in range(len(layer_in.outputs.shape.as_list()))], name=name)

def FunctionLayer(layer_in, name, func):
    n = TileLayer(layer_in,multiples=[1 for _ in range(len(layer_in.outputs.shape.as_list()))], name=name)
    n.outputs = func(n.outputs)
    n.all_layers[-1] = n.outputs
    return n

def simpleResLayer(n, name):
    name = str(name)
    df_dim = n.outputs.shape.as_list()[-1]
    
    nn = Conv2d(n, df_dim, (3, 3), (1, 1), act=leaky_relu_func(), padding='SAME', name=name + '/0')
    #nn = LayerNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h%d/bn1'%(i))
    nn = Conv2d(nn, df_dim, (3, 3), (1, 1), act=None, padding='SAME', name=name + '/1')
    #nn = LayerNormLayer(nn, is_train=is_train, act=lrelu, gamma_init=gamma_init, name='h%d/bn2'%(i))  
    nn.outputs = tl.act.lrelu(nn.outputs + n.outputs, 0.2)
    return nn

def bottleneckResLayer(n, name):
    name = str(name)
    df_dim = n.outputs.shape.as_list()[-1]
    
    nn = Conv2d(n, int(df_dim / 4), (1, 1), (1, 1), act=leaky_relu_func(), padding='SAME', name=name + '/0')
    nn = Conv2d(nn, int(df_dim / 4), (3, 3), (1, 1), act=leaky_relu_func(), padding='SAME', name=name + '/1')
    #nn = LayerNormLayer(nn, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h%d/bn1'%(i))
    nn = Conv2d(nn, df_dim, (3, 3), (1, 1), act=None, padding='SAME', name=name + '/2')
    #nn = LayerNormLayer(nn, is_train=is_train, act=lrelu, gamma_init=gamma_init, name='h%d/bn2'%(i))  
    nn.outputs = tl.act.lrelu(nn.outputs + n.outputs, 0.2)
    return nn

def reset_all():
    tl.layers.clear_layers_name()
    tf.reset_default_graph()
    co = tf.get_collection_ref(tf.GraphKeys.SUMMARIES)
    del co[:]

def place_holder_with_shape(shape, name=None):
    return tf.placeholder(dtype=tf.float32, shape=shape, name=name)

class Model:
    def __init__(self, input_tensor, scope, reuse=False, is_train=True):
        self.input_tensor = input_tensor
        self.scope = scope
        self.reuse = reuse
        self.is_train = is_train
        self.name = 0 # for tensorlayer naming
    
    def __call__(self):
        tl.layers.set_name_reuse(self.reuse)
        with tf.variable_scope(self.scope, reuse=self.reuse) as vs:
            return self.build()
    
    def build(self):
        return tl.layers.InputLayer(self.input_tensor)
    
    def getNewName(self):
        self.name += 1
        return '%04d'%self.name

GENERATOR = 'Generator'
DISCRIMINATOR = 'Discriminator'
ENCODER = 'encoder'
DECODER = 'decoder'

class Generator(Model):
    def __init__(self, input_tensor, scope=GENERATOR, reuse=False, is_train=True):
        Model.__init__(self, input_tensor, scope, reuse, is_train)
        
class Discriminator(Model):
    def __init__(self, input_tensor, scope=DISCRIMINATOR, reuse=False, is_train=True):
        Model.__init__(self, input_tensor, scope, reuse, is_train)
    
def leaky_relu_func(alpha=0.2):
    def leaky_relu(x, alpha=alpha):
        return tl.act.lrelu(x, alpha)
    return leaky_relu

def get_norm_without_batch_axis(x):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(x**2, axis=list(range(len(x.shape)))[1:])))

def get_norm_with_all_axis(x):
    return tf.sqrt(tf.reduce_sum(x**2, axis=list(range(len(x.shape)))))
