import pathlib2
import os
import numpy as np
from skimage.io import imsave
import re

def setGPU(i):
    '''
    if use multi GPU, i is comma seperated, like '1,2,3'
    '''
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(i)
    
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print('gpu(s) to be used: %s'%str(gpus))
    return NGPU

def getHomePath():
    return str(pathlib2.Path.home())

def join_path(a, b):
    return os.path.join(a, b)

def ZipOfPython3(*args):
    '''
    return a iter rather than a list
    each of args should be iterable
    
    =======usage=======
    def f(i):
        while True:
            yield i
            i += 1

    for x in ZipOfPython3(f(1), f(2), f(3)):
        print(x)
    ===== python2 zip will stuck in this infinite generator function, so we should use ZipOfPython3==========
    =====another situation is the function returns a datapoint, and zip will make it a list, meaning the whole dataset will be loaded
        into the memory, which is not desirable=============
    '''
    args = [iter(x) for x in args]
    while True:
        yield [next(x) for x in args]

        
class AccuracyCounter:
    '''
    in supervised learning, we often want to count the test accuracy.
    but the dataset size maybe is not dividable by batch size, causing a remainder fraction which is annoying.
    also, sometimes we want to keep trace with accuracy in each mini-batch(like in train mode)
    this class is a simple class for counting accuracy.
    
    ====usage===
    counter = AccuracyCounter()
    iterate over test set:
        counter.addOntBatch(predict, label) -> return accuracy in this mini-batch
    counter.reportAccuracy() -> return accuracy over whole test set
    '''
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
    def addOntBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self):
        '''
        return nan when 0 / 0
        '''
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)
    
def sphere_sample(size):
    '''
    sample noise from high dimensional gaussian distribution and project it to high dimension sphere of unit ball(with L2 norm = 1)
    '''
    z = np.random.normal(size=size)
    z = z / (np.sqrt(np.sum(z**2, axis=1, keepdims=True)) + 1e-6)
    return z.astype(np.float32)

def sphere_interpolate(a, b, n=64):
    '''
    interpolate along high dimensional ball, according to <Sampling Generative Networks> 
    '''
    a = a / (np.sqrt(np.sum(a**2)) + 1e-6)
    b = b / (np.sqrt(np.sum(b**2)) + 1e-6)
    dot = np.sum(a * b)
    theta = np.arccos(dot)
    mus = [x * 1.0 / (n - 1) for x in range(n)]
    ans = np.asarray([(np.sin((1.0 - mu) * theta) * a + np.sin(mu * theta) * b) / np.sin(theta) for mu in mus], dtype=np.float32)
    return ans

def mergeImage_color(images, rows, cols=None):
    '''
    images:(np.ndarray) 4D array, [N, H, W, C], C = 3
    '''
    cols = rows if not cols else cols
    size = (rows, cols)
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c), dtype=images.dtype) # use images.dtype to keep the same dtype
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def mergeImage_gray(images, rows, cols=None):
    '''
    images:(np.ndarray) 3D array, [N, H, W], C = 3
    '''
    cols = rows if not cols else cols
    size = (rows, cols)
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]), dtype=images.dtype) # use images.dtype to keep the same dtype
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

def to_gray_np(img):
    '''
    normalize a img to lie in the range of [0, 1], turning it into a gray image
    '''
    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min()) * 1).astype(np.float32)
    return img

def to_rgb_np(img):
    '''
    normalize a img to lie in the range of [0, 255], turning it into a RGB image
    '''
    img = img.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img

def getID():
    '''
    return an unique id on each call (useful when you need a sequence of unique identifiers)
    '''
    getID.x += 1
    return getID.x
getID.x = 0

try:
    from IPython.display import clear_output
except ImportError as e:
    pass

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    '''
    change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power) 
    as known as inv learning rate sheduler in caffe, 
    see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

    the default gamma and power come from <Domain-Adversarial Training of Neural Networks>
    
    ====code to see how it changes(decays to %20 at %10 * max_iter under default arg)=======
    from matplotlib import pyplot as plt

    ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
    xs = [x for x in range(10000)]

    plt.plot(xs, ys)
    plt.show()
    
    '''
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter)) ) ** (- power))

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    '''
    change gradually from A to B, according to the formula (from <ImportanceWeighted Adversarial Nets for Partial Domain Adaptation>)
    A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    
    ====code to see how it changes(almost reaches B at %40 * max_iter under default arg)=======
    from matplotlib import pyplot as plt

    ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
    xs = [x for x in range(10000)]

    plt.plot(xs, ys)
    plt.show()
    
    '''
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

def runTask():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxGPU', type=int, default=1000)
    parser.add_argument('--needGPU', type=int, default=1)
    parser.add_argument('--maxLoad', type=float, default=0.1)
    parser.add_argument('--maxMemory', type=float, default=0.1)
    parser.add_argument('--sleeptime', type=float, default=60)
    parser.add_argument('--user', type=str)
    parser.add_argument('file', nargs=1)
    args = parser.parse_args()

    import cPickle
    from subprocess import Popen, PIPE

    import time

    import GPUtil

    import random

    import os
    
    maxGPU = args.maxGPU
    needGPU = args.needGPU
    maxLoad = args.maxLoad
    maxMemory = args.maxMemory
    file = args.file[0]
    user = args.user
    sleeptime = args.sleeptime
    

    while True:
        with open(file) as f:
            lines = [line for line in f if line.strip()]
        if lines:
            while True:
                s = 'for x in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do ps -f -p $x | grep "%s"; done'%user
                p = Popen(s, stdout=PIPE, shell=True)
                ans = p.stdout.read()
                mygpu = len(ans.splitlines())
                deviceIDs = GPUtil.getAvailable(order = 'first', limit = needGPU, maxLoad = maxLoad, maxMemory = maxMemory, includeNan=False, excludeID=[], excludeUUID=[])
                find = False
                if mygpu < maxGPU and len(deviceIDs) >= needGPU:
                    os.system(lines[0].strip())
                    print('runing command(%s)'%lines[0].strip())
                    find = True
                time.sleep(sleeptime)
                if find:
                    break
            with open(file, 'w') as f:
                for line in lines[1:]:
                    f.write(line)
        else:
            break