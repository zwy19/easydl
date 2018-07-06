easydl, easy deeplearning!
===========================

what is easydl
==============

easydl is a python package that aims to ease the development of deep learning algorithms. To install easydl, run ::

    pip install easydl

That's it, super easy!

easydl can do
=====================

easydl mainly contains wrappers and some commonly used
modules. **easydl** hates repeated code(see `DRY <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`_ )
and coupling code.

For example, in **pytorch**, ``optimizer.zero_grad()`` and ``optimizer.step()`` must be called in pair. If we have
several optimizers, code would be like ::

    op1.zero_grad()
    op2.zero_grad()
    op3.zero_grad()

    # do whatever

    op1.step()
    op2.step()
    op3.step()

code like this would be hard to maintain. What if we want to remove one optimizer ? What if we want to add one optimizer?

With **easydl**, code would be like this ::

    with OptimizerManager([op1, op2, op3, op4]):
        # do whatever

The :py:class:`easydl.pytorch.pytorch.OptimizerManager` calls ``zero_grad`` of its arguments when entering the scope,
and calls ``step`` of its arguments when leaving the scope. This way, adding an optimizer and removing an optimizer
become super easy!

easydl can't do
================

easydl doesn't care about namespace pollution. In fact, it makes "namespace pollution" on purpose. It imports all
available functions/classes to the global scope including some commonly used packages.

with one single line ::

    from easydl import *


we don't have to write the following code anymore ::

    # from matplotlib import pyplot as plt
    # import numpy as np
    # import tensorflow as tf
    # import tensorlayer as tl
    # import torch.nn as nn

What's more, all of the functions/classes in easydl have been introduced to the global scope.

get started!
=============

GPU management
--------------

Does this seem familiar to you?

1. run ``nvidia-smi`` to see which GPU is available

#. set ``CUDA_VISIBLE_DEVICES`` to the corresponding id

#. run code

#. find a bug

#. fix it

#. run again

#. GPU out of memory cause one colleague has started an experiment on the same GPU

#. goto step 1

To free developers from tedious work like this, easydl has provided :py:func:`easydl.common.wheel.selectGPUs`. ::

    selectGPUs(1) # automatically select 1 available GPU to run tasks.

GPUs are scarce (although there are some labs with numerous GPUS) and there are always more tasks than GPUs.

Let's say we have 10 tasks and 5 GPUs. We have to run 5 tasks at first, then every one hour or so, we check if any task
has finished. If there are GPUs available, we start a new task until 10 tasks have been finished. The regularly-checking
can be really annoying cause I can't watch a two-hour movie without break.

easydl provides a command :py:func:``easydl.common.commands.runTask``. It will be installed as a command available
within shell after easydl is installed. The basic usage is ::

    runTask tasks.txt --user yourUserName

where we have 10 lines in ``tasks.txt``, each line is a command(a task to run). The magic is that it runs every 60
seconds to see if there is an available GPU to run. If so, it takes one command from ``tasks.txt`` and runs it. Even if
we only have 2 GPUs, we can start this command and watch movie for ten hours without break (don't do it, you'll get
fired).

datasets and dataloaders
-------------------------

log info in training
---------------------

context managers for pytorch
-----------------------------

new nn.Module and functions for pytorch
-----------------------------------------

GAN evaluation metric for tensorflow
-------------------------------------

Accumulator and AccuracyCounter
-----------------------------------


there are more for you to discover!
------------------------------------

we can't list all functions and classes along with its usage in one single page. Try discovering them yourself! The
functions and classes are all fully documented at their respective pages

modules in easydl
===================

- :doc:`common <./modules/easydl.common>`

    this submodule contains functions and classes independent of pytorch and tensorflow. It mainly contains command line
    tools / data preprosessing apis / GPU management / logging and so on.

- :doc:`tf <./modules/easydl.tf>`

    this submodule contains functions and classes special to tensorflow. It mainly contains GAN evaluation metric and
    so on.

- :doc:`pytorch <./modules/easydl.pytorch>`

    this submodule contains functions and classes special to pytorch. It mainly contains some context managers / new
    modules and so on.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`