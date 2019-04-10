#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import logging
import time
import datetime
import os

import config
import data_loader
from lstm_cnn import LSTM_CNN

# Shutup tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Get logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main Process")


# Load data
logger.info("Loading data...")
x_text, y = data_loader.get_dataset(config.POS_FILE, config.NEG_FILE, config.NEU_FILE)
x_test_text, y_test = data_loader.get_dataset(config.POS_TEST_FILE, config.NEG_TEST_FILE, config.NEU_TEST_FILE)

# Build vocabulary
logger.info("Building vocabulary...")
max_document_length = max([len(x.split(" ")) for x in x_text])

vocab, embd = data_loader.load_glove()
vocab_size = len(vocab)
embedding = np.asarray(embd)
embedding_dim = len(embd[0])

W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf)
sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

#init vocab processor
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#fit the vocab from glove
pretrain = vocab_processor.fit(vocab)
#transform inputs
x = np.array(list(vocab_processor.transform(x_text)))
x_test = np.array(list(vocab_processor.transform(x_test_text)))

np.random.seed(42)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(config.DEV_SIZE * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
logger.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = LSTM_CNN(x_train.shape[1], y_train.shape[1], len(vocab_processor.vocabulary_), embedding_dim,
                            config.FILTER_SIZE, config.NUM_FILTERS, config.L2_REG_LAMBDA)
        
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.NUM_CHECKPOINTS)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        #TRAINING STEP
        def train_step(x_batch, y_batch,save=True):
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: config.DROPOUT_PROB
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                train_summary_writer.add_summary(summaries, step)

        #EVALUATE MODEL
        def dev_step(x_batch, y_batch, writer=None,save=True):
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: config.DROPOUT_PROB
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if save:
                if writer:
                    writer.add_summary(summaries, step)
        
        # TEST
        def test_step(x_batch, y_batch):
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: config.DROPOUT_PROB
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


        batches = data_loader.get_batch(list(zip(x_train, y_train)), config.BATCH_SIZE, config.NUM_EPOCHS)

        #TRAIN FOR EACH BATCH
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % config.EVALUATE_EVERY == 0:
                logger.info("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                logger.info("")
            if current_step % config.CHECKPOINT_EVERY == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        test_step(x_test, y_test)

