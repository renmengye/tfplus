import tensorflow as tf
import tfplus
from tfplus.nn import Model
from resnet_imagenet_model_wrapper import ResNetImageNetModelWrapper

NUM_CLS = 1000


class ResNetImageNetModelMultiWrapper(tfplus.nn.ContainerModel):

    def __init__(self, num_replica=2, name='resnet_imagenet_multi_wrapper'):
        super(ResNetImageNetModelMultiWrapper, self).__init__(name=name)
        self.register_option('learn_rate')
        self.register_option('learn_rate_decay')
        self.register_option('steps_per_lr_decay')
        self.register_option('inp_depth')
        self.num_replica = num_replica
        self.set_variable_sharing(True)
        for ii in xrange(num_replica):
            model = tfplus.nn.model.create_from_main(
                'resnet_imagenet_wrapper').set_gpu(ii)
            self.add_sub_model(model)

    def set_gpu(self, value):
        return Model.set_gpu(self, value)

    def build_input(self):
        results = {}
        phase_train = self.add_input_var('phase_train', None, 'bool')
        results['phase_train'] = phase_train
        inp_depth = self.get_option('inp_depth')
        orig_x = []
        for ii in xrange(self.num_replica):
            with tf.name_scope('%s_%d' % ('replica', ii)) as scope:
                device = '/gpu:{}'.format(ii)
                with tf.device(device):
                    x_ = self.add_input_var('x_{}'.format(
                        ii), [None, None, None, inp_depth], 'float')
                    results['x_{}'.format(ii)] = x_
                    y_gt_ = self.add_input_var('y_gt_{}'.format(ii), [
                                               None, NUM_CLS], 'float')
                    results['y_gt_{}'.format(ii)] = y_gt_
                    orig_x.append(
                        (x_ + self.sub_models[0].res_net._img_mean) / 255.0)
                    # self.log.error(x_.device)
        # self.log.fatal('')
        self.register_var('orig_x', tf.concat(0, orig_x))
        return results

    def init_var(self):
        learn_rate = self.get_option('learn_rate')
        lr_decay = self.get_option('learn_rate_decay')
        steps_decay = self.get_option('steps_per_lr_decay')
        learn_rate = tf.train.exponential_decay(
            learn_rate, self.global_step, steps_decay, lr_decay,
            staircase=True)
        self.register_var('learn_rate', learn_rate)
        self.opt = tf.train.MomentumOptimizer(learn_rate, momentum=0.9)
        self.learn_rate = learn_rate
        pass

    def build(self, inp):
        # Divide input equally.
        self.lazy_init_var()
        inp_list = []
        output = []
        for ii in xrange(self.num_replica):
            with tf.name_scope('%s_%d' % ('replica', ii)) as scope:
                device = '/gpu:{}'.format(ii)
                with tf.device(device):
                    tf.get_variable_scope().reuse_variables()
                    inp_ = {
                        'x': inp['x_{}'.format(ii)],
                        'y_gt': inp['y_gt_{}'.format(ii)],
                        'phase_train': inp['phase_train']
                    }
                    output.append(self.sub_models[ii].build(inp_))
                    inp_list.append(inp_)
        self.output_list = output
        self.input_list = inp_list
        output = tf.concat(0, [oo['y_out'] for oo in output])
        self.register_var('y_out', output)
        output2 = tf.concat(0, [mm.get_var('score_out')
                                for mm in self.sub_models])
        self.register_var('score_out', output2)
        return {'y_out': output}

    def build_loss(self, inp, output):
        tower_grads = []
        for ii in xrange(self.num_replica):
            with tf.name_scope('%s_%d' % ('replica', ii)) as scope:
                device = '/gpu:{}'.format(ii)
                with tf.device(device):
                    loss = self.sub_models[ii].build_loss(
                        self.input_list[ii], self.output_list[ii])
                grads = self.opt.compute_gradients(loss)
                self.add_loss(loss)
                tower_grads.append(grads)
        self.tower_grads = tower_grads
        total_loss = self.get_loss()
        self.register_var('loss', total_loss)
        return total_loss

    @staticmethod
    def average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            if grad_and_vars[0][0] is not None:
                for g, _ in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over
                    # below.
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(0, grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
            else:
                average_grads.append(grad_and_vars[0])
        return average_grads

    def build_optim(self, loss):
        global_step = self.global_step
        learn_rate = self.learn_rate
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self.average_gradients(self.tower_grads)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = self.opt.apply_gradients(
            grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            0.999, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # for m in self.sub_models:
        #     self.log.info(m.device)
        # self.log.fatal('haha')
        return train_op

    def build_all(self, param_avg=False):
        """Build all nodes."""
        if self._has_built_all:
            raise Exception('Only call build_all or build_eval once.')
        self._has_built_all = True
        with tf.device('/cpu:0'):
            with tf.variable_scope(self.name):
                inp_var = self.build_input()
                output_var = self.build(inp_var)
                loss_var = self.build_loss(inp_var, output_var)
                train_step = self.build_optim(loss_var)
                self.register_var('train_step', train_step)
        return self

    def restore_weights_from(self):
        self.sub_models[0].restore_weights_from(sess, folder)
        return super(ContainerModel, self).restore_weights_from(sess, folder)
        pass

    def restore_aux_from(self, sess, folder):
        self.sub_models[0].restore_aux_from(sess, folder)
        return super(ContainerModel, self).restore_aux_from(sess, folder)

    def restore_weights_aux_from(self, sess, folder):
        self.sub_models[0].restore_weights_aux_from(sess, folder)
        return super(ContainerModel, self).restore_weights_aux_from(
            sess, folder)

    def get_save_var_list_recursive(self):
        save_vars = []
        save_vars.extend(self.sub_models[0].get_save_var_list_recursive())
        save_vars.extend(self.get_save_var_dict().values())
        return save_vars

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        return results

tfplus.nn.model.register('resnet_imagenet_multi_wrapper',
                         ResNetImageNetModelMultiWrapper)
