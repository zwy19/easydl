import tensorflow as tf
import tensorlayer as tl
import os

def reset_all_summary():
    """
    delete current summaries
    """
    co = tf.get_collection_ref(tf.GraphKeys.SUMMARIES)
    del co[:]

def mergeImageTensor(img, rows, cols=None):
    """
    :param img: img must be 4-D, and batch size should be fixed
    :param rows:
    :param cols:
    :return:
    """
    cols = rows if not cols else cols
    
    imgs = tf.unstack(img, axis=0)
#     imgs = tf.split(img, num_or_size_splits=n * n, axis=0)
    imgs = tf.concat([tf.concat([imgs[i * n + j] for j in range(cols)], axis=0) for i in range(rows)], axis=1)
    return tf.expand_dims(imgs,axis=0)


def summaryScalar(x, name=None):
    tf.summary.scalar(name='scalar_' + (name if name else x.name), tensor=x)


def summaryImage(x, name=None):
    tf.summary.image(name='image_' + (name if name else x.name), tensor=x)


def mergeAllScalars():
    return tf.summary.merge([x for x in tf.get_collection(tf.GraphKeys.SUMMARIES) if 'scalar_' in x.name])


def mergeAllImages():
    return tf.summary.merge([x for x in tf.get_collection(tf.GraphKeys.SUMMARIES) if 'image_' in x.name])


def mergeAllSummary():
    return tf.summary.merge_all()


def getWriter(logdir, clear=False):
    if clear:
        os.system('rm %s -r'%logdir)
    tl.files.exists_or_mkdir(logdir)
    writer = tf.summary.FileWriter(logdir=logdir)
    return writer