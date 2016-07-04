import numpy as np
import tensorflow as tf


class GradientClipOptimizer(tf.train.Optimizer):
    """A optimizer wrapper with gradient clipping."""

    def __init__(self, optimizer, clip=1.0):
        self._optimizer = optimizer
        self._clip = clip

        pass

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        return self._optimizer.apply_gradients(
            grads_and_vars, global_step=global_step, name=name)

    def compute_gradients(self, loss, var_list=None, gate_gradients=1):
        grads_and_vars = self._optimizer.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients)
        results = []
        for grad, var in grads_and_vars:
            # grad, var = pair[0], pair[1]
            if grad is not None:
                grad = tf.clip_by_norm(grad, self._clip)
            results.append((grad, var))
        return results

    def get_slot(self, var, name):
        return self._optimizer.get_slot(var, name)

    def get_slot_name(self):
        return self._optimizer.get_slot_name()

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=1, name=None):
        gvs = self.compute_gradients(loss)
        self.apply_gradients(gvs)
        return self._optimizer.minimize(
            loss, global_step=global_step, var_list=var_list,
            gate_gradients=gate_gradients, name=name)
