import fnmatch
import logger
import os
import yaml
import tensorflow as tf

kMaxToKeep = 2


class Saver():

    def __init__(self, folder, fname='model', var_dict=None):

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.folder = folder
        self.log = logger.get()
        self.fname = fname
        self.tf_saver = None
        if var_dict is None:
            self.var_dict = tf.all_variables()
        else:
            self.var_dict = var_dict
        pass

    def save(self, sess, global_step=None):
        """Save checkpoint.

        Args:
            global_step:
        """
        if self.tf_saver is None:
            self.tf_saver = tf.train.Saver(
                self.var_dict, max_to_keep=kMaxToKeep)
        ckpt_path = os.path.join(self.folder, self.fname + '.ckpt')
        self.log.info('Saving checkpoint to {}'.format(ckpt_path))
        self.tf_saver.save(sess, ckpt_path, global_step=global_step)
        pass

    def get_latest_ckpt(self):
        """Get the latest checkpoint filename in a folder."""

        ckpt_fname_pattern = os.path.join(self.folder, self.fname + '.ckpt-*')
        print ckpt_fname_pattern
        print os.listdir(self.folder)
        ckpt_fname_list = []
        for fn in os.listdir(self.folder):
            fullname = os.path.join(self.folder, fn)
            if fnmatch.fnmatch(fullname, ckpt_fname_pattern):
                if not fullname.endswith('.meta'):
                    ckpt_fname_list.append(fullname)
        if len(ckpt_fname_list) == 0:
            raise Exception('No checkpoint file found.')
        ckpt_fname_step = [int(fn.split('-')[-1]) for fn in ckpt_fname_list]
        latest_step = max(ckpt_fname_step)

        latest_ckpt = os.path.join(self.folder,
                                   self.fname + '.ckpt-{}'.format(latest_step))
        latest_graph = os.path.join(self.folder,
                                    self.fname + '.ckpt-{}.meta'.format(latest_step))
        return (latest_ckpt, latest_graph, latest_step)

    def get_ckpt_info(self):
        """Get info of the latest checkpoint."""

        if not os.path.exists(self.folder):
            raise Exception('Folder "{}" does not exist'.format(self.folder))

        model_id = os.path.basename(self.folder.rstrip('/'))
        self.log.info('Restoring from {}'.format(self.folder))

        ckpt_fname, graph_fname, latest_step = self.get_latest_ckpt()
        self.log.info('Restoring at step {}'.format(latest_step))

        return {
            'ckpt_fname': ckpt_fname,
            'graph_fname': graph_fname,
            'step': latest_step,
            'model_id': model_id
        }

    def restore(self, sess, ckpt_fname=None):
        """Restore the checkpoint file."""
        if ckpt_fname is None:
            ckpt_fname = self.get_latest_ckpt()[0]

        if self.tf_saver is None:
            self.tf_saver = tf.train.Saver(self.var_dict)

        self.tf_saver.restore(sess, ckpt_fname)

        pass
