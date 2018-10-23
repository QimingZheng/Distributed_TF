#!/usr/bin/env python
# coding=utf-8

from __future__ import division

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import argparse
import time
import shutil
import pdb
import tempfile

import tensorflow as tf
from tensorflow.python.client import timeline

from model import BaseModel, DEBUG
from util import get_available_gpus, add_arguments, create_hparams

SINGLE_CARD_SPEED = None
WARM_UP_BATCH = 10


def make_config():
    config = tf.ConfigProto()

    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 56

    return config


def train(config, hparams):
    if hparams.job_name is None or hparams.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if hparams.task_index is None or hparams.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    
    print("job name = %s" % hparams.job_name)
    print("task index = %d" % hparams.task_index)
    
    ps_spec = hparams.ps_hosts.split(",")
    worker_spec = hparams.worker_hosts.split(",")

    num_workers = len(worker_spec)

    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    server = tf.train.Server(
        cluster, job_name=hparams.job_name, task_index=hparams.task_index)

    if hparams.job_name == "ps":
        server.join()
    
    is_chief = (hparams.task_index == 0)

    worker_device = "/job:worker/task:%d/gpu:%d" % (hparams.task_index, 0)

    with tf.device(
        tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:ps/cpu:0",
            cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        model = BaseModel(hparams, mode=tf.contrib.learn.ModeKeys.TRAIN, global_step = global_step)
        print("num_gpus = %d, batch size = %d" %
            (model.num_gpus, hparams.batch_size * model.num_gpus))
        #local_init_op = [model.opt.local_step_init_op]
        #if is_chief:
        #    local_init_op = [model.opt.chief_init_op]
        #local_init_op.extend([model.var_init_op])
        #local_init_op=tf.group(*local_init_op)
        #ready_for_local_init_op = model.opt.ready_for_local_init_op

        #chief_queue_runner = model.opt.get_chief_queue_runner()
        #sync_init_op = model.opt.get_init_tokens_op()
        #init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
        log_dir = tempfile.mktemp()
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=log_dir,
            #ready_for_local_init_op=ready_for_local_init_op,
            #local_init_op=local_init_op,
            saver=model.saver,
            global_step=model.global_step,
            summary_op=None,
            init_op = model.var_init_op,
            save_model_secs=600,
            summary_writer=None,
        )
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sess = sv.prepare_or_wait_for_session(server.target, config=config)
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #if is_chief:
            # Chief worker will start the chief queue runner and call the init op.
        #    sess.run(sync_init_op)
        #    sv.start_queue_runners(sess, [chief_queue_runner])
        #with sess as sess:
        tb_log_dir = "tblog"
        if os.path.exists(tb_log_dir):
            shutil.rmtree(tb_log_dir)
        else:
            os.mkdir(tb_log_dir)
        writer = tf.summary.FileWriter(tb_log_dir, sess.graph)

        pass_id = 0
        batch_id = 0

        total_word_count = 0
        total_batch_id = 0
        start_time = None

        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sess.run(model.var_init_op)
        sess.run(model.iterator.initializer)
        #sess.run(model.table_init_ops)
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        while True:
            try:
                if DEBUG:
                    a, b, c, d, word_count, summaries = sess.run(
                        list(model.fetches.values()) + [model.word_count] +
                        [model.merged_summary_op])
                else:
                    a, b, c, d, word_count, summaries = sess.run(
                        list(model.fetches.values()) + [model.word_count] +
                        [model.merged_summary_op])
                if batch_id == WARM_UP_BATCH:
                    start_time = time.time()
                    total_word_count += word_count if batch_id >= WARM_UP_BATCH else 0
                if batch_id and not batch_id % 5:
                    print("Pass %d, Batch %d " % (pass_id, batch_id),
                        " Loss: ", a)
                batch_id += 1
                total_batch_id += 1
                if batch_id == 5 and pass_id > 0:
                    model.saver.save(
                        sess,
                        "checkpoint/" + "model-%s.ckpt" % str(
                            pass_id
                        ),  # Suggestion: Conduct exp in /data/data1/v-qizhe/ to avoid disk space shortage
                        global_step=batch_id)
                writer.add_summary(summaries, total_batch_id)
                with open("train.log",'a') as file:
                    file.write(str(batch_id)+" "+str(b)+'\n')
            except tf.errors.OutOfRangeError:
                sess.run(model.iterator.initializer)
                batch_id = 0
                pass_id += 1
                continue

def main(unused_argv):
    hparams = create_hparams(FLAGS)
    print(hparams)
    if hparams.mode == "train":
        _mode = tf.contrib.learn.ModeKeys.TRAIN
    elif hparams.mode == "eval":
        _mode = tf.contrib.learn.ModeKeys.EVAL
    elif hparams.mode == "infer":
        _mode = tf.contrib.learn.ModeKeys.INFER
    else:
        raise("Unknown Mode!!!")

    #model = BaseModel(hparams, mode=_mode)
    config = make_config()
    if _mode == tf.contrib.learn.ModeKeys.TRAIN:
        train(config, hparams)
    else:
        raise(NotImplementedError)
    """
    if _mode == tf.contrib.learn.ModeKeys.EVAL:
        eval(model, config, hparams)
    if _mode == tf.contrib.learn.ModeKeys.INFER:
        infer(model, config, hparams)
    """


if __name__ == "__main__":
    param_parser = argparse.ArgumentParser()
    add_arguments(param_parser)
    FLAGS, unparsed = param_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
