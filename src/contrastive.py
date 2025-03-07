import tensorflow as tf


def contrastive_loss(logits1, logits2, label, margin, eps=1e-7):
    Dw = tf.sqrt(eps + tf.reduce_sum(tf.square(logits1 - logits2), 1))
    loss = tf.reduce_mean((1. - tf.cast(label, tf.float32)) * tf.square(Dw) + tf.cast(label, tf.float32) * tf.square(
        tf.math.maximum(margin - Dw, 0)))
    return loss, Dw

tf.get_logger()