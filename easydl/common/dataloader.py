import os
import numpy as np
import tensorpack
import time
import random
import numbers
from scipy.misc import imread, imresize
import tensorlayer as tl
import cPickle
from wheel import *

import warnings
# disable warning of imread
warnings.filterwarnings('ignore', message='.*', category=Warning)


class CustomDataLoader(object):
    """dataloader to load datapoints of dataset and merge them as a mini-batch

    **usage**:

    1. train mode(the data is random, and different in each call of dl.generator())::

        ds = TestDataset(is_train=True)
        dl = CustomDataLoader(dataset=ds, batch_size=6, num_threads=2)
        for (x, y) in dl.generator():
            print((x, y, x.shape, y.shape))
        for (x, y) in dl.generator():
            print((x, y, x.shape, y.shape))

    2. test mode(the data has a specific order, each call of dl.generator() returns the same data sequence)::

        ds = TestDataset(is_train=False)
        dl = CustomDataLoader(dataset=ds, batch_size=6, num_threads=2)
        for (x, y) in dl.generator():
            print((x, y, x.shape, y.shape))
        assert x.shape[0] == ds.N % dl.batch_size
        for (x, y) in dl.generator():
            print((x, y, x.shape, y.shape))
    """

    
    def __init__(self, dataset, batch_size, num_threads=8,remainder=None):
        """
        :param dataset: subclass of tensorpack.dataflow.RNGDataFlow

        :param int batch_size: batch size

        :param int num_threads:
            numbers of threads to use (training mode is inferred according to dataset,
            if not training, this argument is ignored)

        :param bool remainder:
            **useful only in test mode, and you'd better not set this arg cause it can be inferred from attributes of dataset arg**
            whether to use data that can't form a whole batch, like you have 100 data, batchsize is 3, this arg indicates
            whether to use the remaining 1 data
        """

        self.ds0 = dataset
        self.batch_size = batch_size
        self.num_threads = num_threads
        
        if not remainder:
            try:
                is_train = self.ds0.is_train
                remainder = False if is_train else True # if is_train, there is no need to set reminder 
            except Exception as e:
                # self.ds0 maybe doesn't have is_train attribute, then it has no test mode, set remainder = False
                remainder = False
        
        # use_list=False, for each in data point, add a batch dimension (return in numpy array)
        self.ds1 = tensorpack.dataflow.BatchData(self.ds0, self.batch_size,remainder=remainder, use_list=False,) 
        
        # use 1 thread in test to avoid randomness (test should be deterministic)
        self.ds2 = tensorpack.dataflow.PrefetchDataZMQ(self.ds1, nr_proc=self.num_threads if not remainder else 1)
        
        # required by tensorlayer package
        self.ds2.reset_state()
    
    def generator(self):
        """
        :return: if ``self.ds0.get_data()`` returns N elements,
            then this function returns a generator, which yields N elements in total (actually, it depends on ``ds0.size()``)
        """
        return self.ds2.get_data()


class BaseDataset(tensorpack.dataflow.RNGDataFlow):
    """
    base class for dataset

    what the subclass should do:

    #. remember arguments in ``__init__``, call ``super.__init__``

    #. implement ``_fill_data`` : fill content for ``self.datas`` and ``self.labels``

    #. implement ``_get_one_data``

    the pipeline looks like this:

    #. in ``__init__``, ``_fill_data`` is called to load datas and labels (at this point, data may be a file path)

    #. ``skip_pred`` is used to filter out some data

    #. read a data (usually read from a file path to get the image) by calling ``_get_one_data``,
        pass it to ``transform``, and return the results
    """
    def __init__(self, is_train=True, skip_pred=None, transform=None):
        """
        :param bool is_train: train mode or test mode
        :param skip_pred: predicate with signature of ``(data, label, is_train) -> bool`` . each data will be passed into
            this predicate , data that makes this predicate returns True will be omitted (**can be used to filter
            out some data**). the predicate needs the argument ``is_train`` in case that one may want to use
            different filtering strategy in test and train mode.
        :param transform: predicate with signature of ``(data, label, is_train) -> (data,label)`` . can be used to perform
            some preprocessing task like image resize or so. the predicate needs the argument ``is_train``
            in case that one may want to use different transform strategy in test and train mode (for example, random
            cropping in training and deterministic cropping in test). the predicate needs the argument ``label``
            so that users can have complete control (maybe some transform needs the label information?)
        """
        self.is_train = is_train
        self.skip_pred = skip_pred or (lambda data, label, is_train : False)
        self.transform = transform or (lambda data, label, is_train : (data, label))

        self.datas = []
        self.labels = []

        self._fill_data()

        self._post_init()

    def _fill_data(self):
        """
        should be implemented by subclass. fill content for ``self.datas`` and ``self.labels``
        """
        raise NotImplementedError("not implemented!")

    def _post_init(self):
        """
        filter out some data that makes skip_pred return True
        """
        tmp = [[data, label]  for (data, label) in zip(self.datas, self.labels) if not self.skip_pred(data, label, self.is_train) ]
        self.datas = [x[0] for x in tmp]
        self.labels = [x[1] for x in tmp]

    def size(self):
        return len(self.datas)

    def _get_one_data(self, data, label):
        """
        should be implemented by subclass
        """
        raise NotImplementedError("not implemented!")

    def get_data(self):
        """
        returned data and label should be at least rank 1, but can't be rank 0 (just a number)
        """
        size = self.size()
        ids = list(range(size))
        for _ in range(size):
            id = random.choice(ids) if self.is_train else _
            data, label = self._get_one_data(self.datas[id], self.labels[id])
            data, label = self.transform(data, label, self.is_train)
            yield np.asarray(data), np.asarray([label]) if isinstance(label, numbers.Number) else label


class TestDataset(BaseDataset):
    """
    simple test dataset to store N data, ith data is ``([i, i+1], [2i+1])`` where 0 <= i < N

    usage::

        for (x, y) in TestDataset(is_train=True).get_data():
            print((x, y))
        for (x, y) in TestDataset(is_train=False).get_data():
            print((x, y))
    """
    def __init__(self, N=100,  is_train=True, skip_pred=None, transform=None):
        self.N = N
        super(TestDataset, self).__init__(is_train=is_train, skip_pred=skip_pred, transform=transform)

    def _fill_data(self):
        self.datas = [[i, i + 1] for i in range(self.N)]
        self.labels = [2 * i + 1 for i in range(self.N)]

    def _get_one_data(self, data, label):
        return data, label


class CombinedDataset(BaseDataset):
    """
    combine multiple datasets to get a new dataset.

    this can be useful if one wants to combine different datasets with various size in training (the ``weights``
    argument can be used to achieve balance between datasets)

    this class concatenates all data in ``datasets``, and generates data points randomly from one dataset
    according to the given weight
    """
    def __init__(self, datasets, weights):
        """
        :param list datasets: list of datasets to be combined
        :param list weights:  list of weight for each datasets
        """
        self.datasets = datasets
        self.weights = weights
        super(CombinedDataset, self).__init__(is_train=True, skip_pred=None, transform=None)

    def _fill_data(self):
        self.datas = sum([x.datas for x in self.datasets], [])
        self.labels = sum([x.labels for x in self.datasets], [])

        # make weights a probability distribution
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.weights = self.weights / np.sum(self.weights)

        self.iters = [x.get_data() for x in self.datasets]
        self.indexes = np.asarray(list(range(len(self.datasets))), dtype=np.int)

    def _get_one_data(self, data, label):
        index = np.random.choice(self.indexes, p=self.weights)
        try:
            return next(self.iters[index])
        except StopIteration as e:
            self.iters[index] = self.datasets[index].get_data()
            return next(self.iters[index])


class BaseImageDataset(BaseDataset):
    """
    base image dataset

    for image dataset, ``_get_one_data`` usually reads image from file path

    by default, if the data is colored image, then it's in ``uint8`` type; if it's gray image, then it's in
    ``float`` type.

    by default, the returned label is int. to make the label one-hot, one should do it in ``transform``
    """
    def __init__(self, imsize=224, is_train=True, skip_pred=None, transform=None):
        self.imsize = imsize
        super(BaseImageDataset, self).__init__(is_train, skip_pred, transform)

    def _get_one_data(self, data, label):
        im = imread(data, mode='RGB')
        im = imresize(im, (self.imsize, self.imsize))
        return im, label


def one_hot(n_class, index):
    """
    make an one-hot label
    """
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp


class FileListDataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :

    image_path label_id
    image_path label_id
    ......

    i.e., each line contains an image path and a label id
    """
    def __init__(self, list_path, path_prefix='', imsize=224, is_train=True, skip_pred=None, transform=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        """
        self.list_path = list_path
        self.path_prefix = path_prefix

        super(FileListDataset, self).__init__(imsize=imsize, is_train=is_train, skip_pred=skip_pred, transform=transform)

    def _fill_data(self):
        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1]] for line in f.readlines() if line.strip()] # avoid empty lines
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is space in image path?')
                raise e


class UnLabeledImageDataset(BaseImageDataset):
    """
    applies to image dataset in one directory without labels for unsupervised learning, like getchu, celeba etc

    there is no test mode in unsupervised learning

    **although this is UnLabeledImageDataset, it returns useless labels to have similar interface with other datasets**
    """

    def __init__(self, root_dir,imsize=128, is_train=True, skip_pred=None, transform=None):
        """

        :param root_dir:  search ``root_dir`` recursively for all files (treat all files as image files)
        """
        self.root_dir = root_dir
        super(UnLabeledImageDataset, self).__init__(imsize, is_train, skip_pred, transform)

    def _fill_data(self):
        self.datas = sum(
            [[os.path.join(path, file) for file in files] for path, dirs, files in os.walk(self.root_dir) if files], [])
        self.labels = [0 for x in self.datas] # useless label


class ImageFolderDataset(BaseImageDataset):
    """
    dataset for specific directory hierachy::

        root_dir
            class1
                file1
                file2
                ...
            class2
                file1
                file2
                ...
            ...

    class names are collected and sorted alphabetically , then converted to integers.

    to convert between class names and integers, use ``NameToId`` and ``IdToName`` properties
    """
    def __init__(self, root_dir,imsize=128, is_train=True, skip_pred=None, transform=None):
        self.root_dir = root_dir
        super(ImageFolderDataset, self).__init__(imsize, is_train, skip_pred, transform)

    def _fill_data(self):
        self.datas = sum([[os.path.join(path, file) for file in files] for path, dirs, files in os.walk(self.root_dir) if files], [])
        self.labels = [file.split(os.sep)[-2] for file in self.datas]
        self.classes = sorted(list(set(self.labels)))
        self.NameToId = {x : i for (i, x) in enumerate(self.classes)}
        self.IdToName = {i : x for (i, x) in enumerate(self.classes)}

        self.labels = [self.NameToId[x] for x in self.labels]


from tensorflow.examples.tutorials.mnist import input_data


class MNISTDataset(BaseDataset):
    def __init__(self, root_dir, is_train=True, skip_pred=None, transform=None):
        """
        :param root_dir: directory that contains **train-images-idx3-ubyte.gz** etc.
        """
        self.root_dir = root_dir
        super(MNISTDataset, self).__init__(is_train, skip_pred, transform)

    def _fill_data(self):
        self.mnist = input_data.read_data_sets(train_dir=self.root_dir, one_hot=False)
        self.current_ds = self.mnist.train if self.is_train else self.mnist.test
        self.datas = self.current_ds.images
        self.labels = self.current_ds.labels

    def _get_one_data(self, data, label):
        return data.reshape((28, 28, 1)), label

    
class Cifar10Dataset(BaseDataset):
    def __init__(self, root_dir, is_train=True, skip_pred=None, transform=None):
        """
        :param root_dir: directory that contains **cifar10 directory with cifar-10-python.tar.gz in it**

        the hierarchy looks like this::

        root_dir
        |__cifar10
            |_cifar-10-python.tar.gz

        """
        self.root_dir = root_dir
        super(Cifar10Dataset, self).__init__(is_train, skip_pred, transform)

    def _fill_data(self):
        self.x_train, self.y_train, self.x_test, self.y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3),
                                                                                             path=self.root_dir)

        self.current_x = self.x_train if self.is_train else self.x_test
        self.current_y = self.y_train if self.is_train else self.y_test
        self.datas = self.current_x
        self.labels = self.current_y

    def _get_one_data(self, data, label):
        return data, label


class Cifar100Dataset(BaseDataset):
    def __init__(self, root_dir, is_train=True, skip_pred=None, transform=None):
        """
        :param root_dir: directory that contains cifar-100 data

        the hierarchy looks like this::

        root_dir
        |__cifar-100-python
            |_test
            |_train

        """
        self.root_dir = root_dir
        super(Cifar100Dataset, self).__init__(is_train, skip_pred, transform)

    def _fill_data(self):
        self.cifar100 = cPickle.load(open(os.path.join(self.root_dir, 'cifar-100-python/train'), 'rb'))
        self.x_train = self.cifar100['data']
        self.x_train.resize((self.x_train.shape[0], 32, 32, 3))
        self.y_train = self.cifar100['fine_labels']

        self.cifar100_test = cPickle.load(open(os.path.join(self.root_dir, 'cifar-100-python/test'), 'rb'))
        self.x_test = self.cifar100_test['data']
        self.x_test.resize((self.x_test.shape[0], 32, 32, 3))
        self.y_test = self.cifar100_test['fine_labels']

        self.current_x = self.x_train if self.is_train else self.x_test
        self.current_y = self.y_train if self.is_train else self.y_test
        self.datas = self.current_x
        self.labels = self.current_y

    def _get_one_data(self, data, label):
        return data, label