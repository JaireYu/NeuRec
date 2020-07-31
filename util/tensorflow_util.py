__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "euclidean_distance", "l2_distance",
           "l2_loss", "get_variable",
           "square_loss", "sigmoid_cross_entropy", "pointwise_loss",
           "log_sigmoid", "bpr_loss", "hinge", "pairwise_loss"]

import tensorflow as tf
from reckit import typeassert
from collections import OrderedDict
from util.common_util import InitArg
from util.common_util import Reduction
from functools import partial


@typeassert(init_method=str, trainable=bool, name=(str, None))
def get_variable(shape, init_method, trainable=True, name=None):
    initializers = OrderedDict()
    initializers["normal"] = tf.initializers.random_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
    initializers["truncated_normal"] = tf.initializers.truncated_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
    initializers["uniform"] = tf.initializers.random_uniform(minval=InitArg.MIN_VAL, maxval=InitArg.MAX_VAL)
    initializers["he_normal"] = tf.initializers.he_normal()
    initializers["he_uniform"] = tf.initializers.he_uniform()
    initializers["xavier_normal"] = tf.initializers.glorot_normal()
    initializers["xavier_uniform"] = tf.initializers.glorot_uniform()
    initializers["zeros"] = tf.initializers.zeros()
    initializers["ones"] = tf.initializers.ones()

    if init_method not in initializers:
        init_list = ', '.join(initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    init = initializers[init_method](shape=shape, dtype=tf.float32)

    return tf.Variable(init, trainable=trainable, name=name)


def _reduce_loss(loss, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    if reduction == Reduction.SUM:
        loss = tf.reduce_sum(loss)
    elif reduction == Reduction.MEAN:
        loss = tf.reduce_mean(loss)
    elif reduction == Reduction.NONE:
        pass

    return loss


def square_loss(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    loss = tf.squared_difference(y_pre, y_true)
    return _reduce_loss(loss, reduction)


def sigmoid_cross_entropy(y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pre)
    return _reduce_loss(loss, reduction)


@typeassert(loss=str, reduction=str)
def pointwise_loss(loss, y_pre, y_true, reduction=Reduction.SUM):
    Reduction.validate(reduction)

    losses = OrderedDict()
    losses["square"] = square_loss
    losses["sigmoid_cross_entropy"] = sigmoid_cross_entropy

    if loss not in losses:
        loss_list = ', '.join(losses.keys())
        ValueError(f"'loss' is invalid, and must be one of '{loss_list}'")

    return losses[loss](y_pre, y_true, reduction=reduction)


def log_sigmoid(y_diff, reduction=Reduction.SUM):
    """bpr loss
    """
    Reduction.validate(reduction)

    loss = -tf.log_sigmoid(y_diff)
    return _reduce_loss(loss, reduction)


bpr_loss = log_sigmoid


def hinge(y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)
    loss = tf.nn.relu(1.0-y_diff)
    return _reduce_loss(loss, reduction)


@typeassert(loss=str, reduction=str)
def pairwise_loss(loss, y_diff, reduction=Reduction.SUM):
    Reduction.validate(reduction)

    losses = OrderedDict()
    losses["log_sigmoid"] = log_sigmoid
    losses["bpr"] = bpr_loss
    losses["hinge"] = hinge
    losses["square"] = partial(square_loss, y_true=1.0)

    if loss not in losses:
        loss_list = ', '.join(losses.keys())
        ValueError(f"'loss' is invalid, and must be one of '{loss_list}'")

    return losses[loss](y_diff, reduction=reduction)


def inner_product(a, b):
    return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def euclidean_distance(a, b):
    return tf.norm(a - b, ord='euclidean', axis=-1)


l2_distance = euclidean_distance


def l2_loss(*params):
    """L2 loss

    Compute  the L2 norm of tensors without the `sqrt`:

        output = sum([sum(w ** 2) / 2 for w in weights])

    Args:
        *weights: Variable length weight list.

    """
    return tf.add_n([tf.nn.l2_loss(w) for w in params])
