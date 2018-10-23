from six.moves import xrange

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes

from util import *
from data_reader import *
from seq2seq_model import seq2seq_model
from BasicSeq2Seq import Seq2SeqModel

DEBUG = False


class BaseModel(object):
    def __init__(self,
                 hparams,
                 mode=tf.contrib.learn.ModeKeys.TRAIN,
                 worker_prefix="",
                 global_step=None):
        self.params = hparams
        self.num_gpus = 1

        self.iterator, self.tgt_bos_id, self.tgt_eos_id = self.get_input_iterator(
            hparams)
        self.word_count = tf.reduce_sum(
            self.iterator.source_sequence_length) + tf.reduce_sum(
                self.iterator.target_sequence_length)

        self.mode = mode
        self.model = Seq2SeqModel(mode, self.tgt_bos_id, self.tgt_eos_id)
        # self.model = seq2seq_model(mode, self.tgt_bos_id, self.tgt_eos_id)

        self.source = self.iterator.source
        self.target_input = self.iterator.target_input
        self.target_output = self.iterator.target_output
        self.source_sequence_length = self.iterator.source_sequence_length
        self.target_sequence_length = self.iterator.target_sequence_length

        self.batch_size = tf.reduce_sum(self.target_sequence_length)

        self.param_server_device = hparams.param_server_device
        self.local_parameter_device = hparams.local_parameter_device
        #self.global_step_device = self.cpu_device

        self.infer_helper = None
        self.debug_helper = None
        self.fetches = {}

        # with tf.device(tf.train.replica_device_setter(cluster=cluster, 
        #     worker_device = '/job:worker/task%d/cpu:0' % hparams.task_index)):
        
        self.logits, self.loss, self.final_context_state = self.model.build_model(
            hparams=hparams,
            source=self.source[0],
            target_input=self.target_input[0],
            target_output=self.target_output[0],
            source_sequence_length=self.source_sequence_length[0],
            target_sequence_length=self.target_sequence_length[0],
        )

        self.fetches["loss"] = self.loss
        self.fetches["logits"] = self.logits
        self.fetches["final_context_state"] = self.final_context_state
        self.global_step = global_step #tf.Variable(0, name="global_step", trainable=False)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.opt = tf.train.AdamOptimizer(self.params.learning_rate)
            #self.opt = tf.train.SyncReplicasOptimizer(
            #    self.opt,
            #    replicas_to_aggregate=1,#self.params.num_workers,
            #    total_num_replicas=1,#self.params.num_workers,
            #    name="sync_replicas")
            self.train_step = self.opt.minimize(self.loss, global_step=self.global_step)
            self.fetches["train_op"] = self.train_step
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN and DEBUG:
            #self.fetches["grads"] = self.debug_helper
            self.fetches["src_len"] = self.source_sequence_length
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.fetches["logits"] = self.infer_helper
            self.fetches["answer"] = self.target_output
            self.fetches["seq_len"] = self.target_sequence_length
        #fetches_list = nest.flatten(list(self.fetches.values()))
        #self.main_fetch_group = tf.group(*fetches_list)
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.merged_summary_op = tf.summary.merge_all()
        local_var_init_op = tf.local_variables_initializer()
        global_var_init_op = tf.global_variables_initializer()
        table_init_ops = tf.tables_initializer()
        variable_mgr_init_ops = [local_var_init_op,global_var_init_op]
        if table_init_ops:
            variable_mgr_init_ops.extend([table_init_ops])
        with tf.control_dependencies([local_var_init_op]):
            variable_mgr_init_ops.extend([])
        self.var_init_op = tf.group(*variable_mgr_init_ops)
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    def get_input_iterator(self, hparams):
        return get_iterator(
            src_file_name=hparams.src_file_name,
            tgt_file_name=hparams.tgt_file_name,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            batch_size=hparams.batch_size,
            num_splits=self.num_gpus,
            disable_shuffle=hparams.disable_shuffle,
            output_buffer_size=self.num_gpus * 1000 * hparams.batch_size,
            num_buckets=hparams.num_buckets,
        )