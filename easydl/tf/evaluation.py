from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy.misc import imread
from scipy import linalg
import pathlib2 as pathlib
import urllib

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys


def create_inception_graph(pth):
    """
    Creates a graph from saved GraphDef file.
    for inception net, it is usually file path to **classify_image_graph_def.pb**
    """
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='')

    # make it usable for any batch size (originally, it can only accept batch size of 1)
    ops = tf.get_default_graph().get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)


def get_activations(images, sess, batch_size=50, verbose=False):
    """
    Calculates the activations of the pool_3 layer for all images.
    :param images: Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    :param sess: current session
    :param batch_size: the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    :param verbose: If set to True and parameter out_step is given, the number of calculated
                     batches is reported(printed out).
    :return: A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = sess.graph.get_tensor_by_name('pool_3:0')
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches))
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    if verbose:
        print(" done")
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    :param mu1: Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    :param sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    :param mu2: The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    :param sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    :param eps:
    :return: The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def _handle_path(path):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    x = np.array([imread(str(fn),mode='RGB').astype(np.float32) for fn in files])
    return x


def get_fid(inputs, inception_path, verbose=True, batch_size=64):
    """
    calculate **FID**(see *GANs trained by a two time-scale update rule converge to a Nash equilibrium* for details)

    code mainly comes from https://github.com/bioinf-jku/TTUR with some revision.

    :param inputs: can be [images(np.ndarray), images(np.ndarray)], or [/path/to/images, /path/to/images].
            **np.ndarray should lie in 0 and 256!**
    :param inception_path: for inception net, it is usually file path to **classify_image_graph_def.pb**
        inception model can be downloaded at http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    :param verbose: If set to True and parameter out_step is given, the number of calculated
                     batches is reported(printed out).
    :param batch_size: the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    :return: FID of two inputs.
    """
    if isinstance(inputs[0], np.ndarray) and isinstance(inputs[1], np.ndarray):
        assert inputs[0].min() >= 0 and inputs[0].max() >= 10 and inputs[0].max() <= 256  # range check
    else:
        inputs[0] = _handle_path(inputs[0])
        inputs[1] = _handle_path(inputs[1])

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        act1 = get_activations(inputs[0], sess, batch_size, verbose)
        act2 = get_activations(inputs[1], sess, batch_size, verbose)
        m1, s1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
        m2, s2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


def get_inception_score(path_or_images, inception_path, batch_size=64, splits=10):
    """
    calculate inception score (see *Improved techniques for training gans* for details)

    :param path_or_images: can be images(np.ndarray), or /path/to/images.
            **np.ndarray should lie in 0 and 256!**
    :param inception_path: for inception net, it is usually file path to **classify_image_graph_def.pb**
        inception model can be downloaded at http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    :param batch_size: the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    :param splits: split and calculate multiple times to get mean and std for inception score
    :return: mean and std for inception score
    """
    if not isinstance(path_or_images, np.ndarray):
        path_or_images = _handle_path(path_or_images)

    images = path_or_images

    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = batch_size

    create_inception_graph(inception_path)
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        w = sess.graph.get_tensor_by_name('softmax/weights_1:0')
        b = sess.graph.get_tensor_by_name('softmax/biases_1:0')
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w, ) + b
        softmax = tf.nn.softmax(logits)

        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


__all__ = ['get_fid', 'get_inception_score']