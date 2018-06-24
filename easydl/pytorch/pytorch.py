import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models
import torchvision.utils as vutils
from collections import Iterable
import math

EPSILON = 1e-20

def weight_init(m):
    """
    initialize weight in neural nets
    use MSRA initializer for conv2d (see <Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification>)
        and use xavier initializer for Linear (see <Understanding the difficulty of training deep feedforward neural networks>)

    usage::

        m.apply(weight_init)

    :param m: input module
    :return:
    """
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.in_features + m.out_features
        m.weight.data.normal_(0.0, math.sqrt(2.0 / n))
        try: # maybe bias=False
            m.bias.data.zero_()
        except Exception as e:
            pass


class TorchReshapeLayer(nn.Module):
    """
    make reshape operation a module that can be used in ``nn.Sequential``
    """
    def __init__(self, shape_without_batchsize):
        super(TorchReshapeLayer, self).__init__()
        self.shape_without_batchsize = shape_without_batchsize

    def forward(self, x):
       x = x.view(x.size(0), *self.shape_without_batchsize)
       return x


class TorchIdentityLayer(nn.Module):
    def __init__(self):
        super(TorchIdentityLayer, self).__init__()

    def forward(self, x):
       return x


class TorchLeakySoftmax(nn.Module):
    """
    leaky softmax, x_i = e^(x_i) / (sum_{k=1}^{n} e^(x_k) + coeff) where coeff >= 0

    usage::

        a = torch.zeros(3, 9)
        TorchLeakySoftmax().forward(a) # the output probability should be 0.1 over 9 classes

    """
    def __init__(self, coeff=1.0):
        super(TorchLeakySoftmax, self).__init__()
        self.coeff = coeff
        
    def forward(self, x):
        x = torch.exp(x)
        x = x / (torch.sum(x, dim=-1, keepdim=True) + self.coeff)
        return x, torch.sum(x, dim=-1, keepdim=True)


class TorchRandomProject(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(TorchRandomProject, self).__init__()
        self.register_buffer('matrix', Variable(torch.randn(1, out_dim, input_dim)))
    def forward(self, x):
        x = x.resize(x.size(0), 1, x.size(1))
        x = torch.sum(self.matrix * x, dim=-1)
        return x    


class GradientReverseLayer(torch.autograd.Function):
    """
    usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::

        x = Variable(torch.ones(1, 2), requires_grad=True)
        grl = GradientReverseLayer()
        grl.coeff = 0.5
        y = grl(x)

        y.backward(torch.ones_like(y))

        print(x.grad)

    """
    def __init__(self):
        self.coeff = 1.0
    def forward(self, input):
        return input
    def backward(self, gradOutput):
        return -self.coeff * gradOutput


class GradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

    usage::

        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))

        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.grl = GradientReverseLayer()
    def forward(self, x):
        coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        self.grl.coeff = coeff
        return self.grl(x)


class OptimizerManager:
    """
    automatic call op.zero_grad() when enter, call op.step() when exit
    usage::

        with OptimizerManager(op): # or with OptimizerManager([op1, op2])
            b = net.forward(a)
            b.backward(torch.ones_like(b))

    """
    def __init__(self, optims):
        self.optims = optims if isinstance(optims, Iterable) else [optims]
    def __enter__(self):
        for op in self.optims:
            op.zero_grad()
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
    
    
class OptimWithSheduler:
    """
    usage::

        op = optim.SGD(lr=1e-3, params=net.parameters()) # create an optimizer
        scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=100, power=1, max_iter=100) # create a function
        that receives two keyword arguments:step, initial_lr
        opw = OptimWithSheduler(op, scheduler) # create a wrapped optimizer
        with OptimizerManager(opw): # use it as an ordinary optimizer
            loss.backward()
    """
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1
    
class TrainingModeManager:
    """
    automatic set and reset net.train(mode)
    usage::

        with TrainingModeManager(net, train=True): # or with TrainingModeManager([net1, net2], train=True)
            do whatever
    """
    def __init__(self, nets, train=False):
        self.nets = nets if isinstance(nets, Iterable) else [nets]
        self.modes = [net.training for net in nets]
        self.train = train
    def __enter__(self):
        for net in self.nets:
            net.train(self.train)
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for (mode, net) in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
    
def variable_to_numpy(x):
    """
    convert a variable to numpy, avoid too many parenthesis
    if the variable only contain one element, then convert it to python float(usually this is the test/train/dev accuracy)
    :param x:
    :return:
    """
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        return float(ans)
    return ans

def merge_ncwh_to_one_image(x):
    """
    :param x: a variable contains image with NCWH format
    :return: a numpy image with shape [1, H, W, C]
    """
    try:
        # maybe min = max, zero division
        nrow = int(math.ceil(x.size(0) ** 0.5))
        x = vutils.make_grid(x.data, nrow=nrow, padding=0, normalize=True) # torch.cuda.FloatTensor, [3, 224, 224]
        x.unsqueeze_(-1)
        x = x.permute(3, 1, 2, 0)
        return x.cpu().numpy()
    except Exception as e:
        return None

def addkey(diction, key, global_vars):
    '''
    add a Variable to log, reduce replicate names like d['a'] = a
    '''
    diction[key] = global_vars[key]
    
def track_scalars(logger, names, global_vars):
    """
    track scalar variables by their names.
    :param logger:
    :param names: variable names to log
    :param global_vars: ``globals()`` or ``locals()``
    """
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = variable_to_numpy(values[k])
    for k, v in values.items():
        logger.log_scalar(k, v)
    logger.step += 1
    print(values)

def track_images(logger, names, global_vars):
    """
    track image variables by their names.
    :param logger:
    :param names: variable names to log
    :param global_vars: ``globals()`` or ``locals()``
    """
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = merge_ncwh_to_one_image(values[k])
        if values[k] is not None:
            logger.log_images(k, values[k])
        else:
            print('images generated are of the same value!')

def post_init_module(module):
    '''
    after initialize a nn.Module, initialize its weight and place it on cuda
    return disposed module, to support x = post_init_module(SomeLayer())
    '''
    module.apply(weight_init)
    return module.cuda()
    
def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    """
    binary cross entropy for multi classification

    label and predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    code to play around::

        N = 100
        C = 20
        x = Variable(torch.ones(N, C))
        prob = nn.functional.softmax(x, dim=-1)
        out = BCELossForMultiClassification(label=x, predict_prob=prob)
        print(out)

        iw = Variable(torch.zeros(N))
        iw[3] = 1
        out = BCELossForMultiClassification(label=x, predict_prob=prob, instance_level_weight=iw)
        print(out)

        cw = Variable(torch.zeros(C))
        cw[3] = 1
        out = BCELossForMultiClassification(label=x, predict_prob=prob, class_level_weight=cw)
        print(out)

    :param label:
    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = label.size()
    N_, C_ = predict_prob.size()
    
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label*torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    """
    entropy for multi classification

    predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = predict_prob.size()
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    """
    cross entropy for multi classification

    label and predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    :param label:
    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = label.size()
    N_, C_ = predict_prob.size()
    
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)