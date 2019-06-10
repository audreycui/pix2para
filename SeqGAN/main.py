#!/usr/bin/python
import tensorflow as tf

from config import Config
from SeqGAN import SeqGAN
from model_gan2 import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import numpy as np

from utils.misc import ImageLoader
import sys
import os

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If specified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

tf.flags.DEFINE_string('image_file','./man.jpg','The file to test the CNN')

tf.flags.DEFINE_string('model', 'SeqGAN',
                       'model type, can be SeqGAN, Show&Tell, and others to be added')


def main(argv):
    config = Config()

    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size
    config.trainable_variable = FLAGS.train_cnn

    np.random.seed(config.seed)
    tf.random.set_random_seed(config.seed)

    if FLAGS.model == 'SeqGAN':
        model = SeqGAN(config)
    elif FLAGS.model == 'Show&Tell':
        model = CaptionGenerator(config)
    else:
        model = SeqGAN(config)

    with tf.Session() as sess:
        print("FLAG: " + FLAGS.phase)
        if FLAGS.phase == 'train':
            # training phase
            #image_path = 'D:/download/COCO/train/images/COCO_train2014_000000318556.jpg'
            #image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')

            #data = prepare_train_data(config) 
            
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            #load the cnn file
            #if FLAGS.load_cnn:
                #model.load_cnn(sess, FLAGS.cnn_model_file)
                #model.load_vgg()
            #tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.phase == 'eval':
            print("EVALUATING")
            # evaluation phase
            eval_data= prepare_eval_data(config)
            #model.load(sess, FLAGS.model_file)
            #sess.run(tf.global_variables_initializer())
            #tf.get_default_graph().finalize()
            #saver = tf.train.Saver(tf.global_variables())
            #saver.restore(sess,FlAGS.model_file)
            model.eval(sess, eval_data)

        elif FLAGS.phase == 'test_loaded_cnn':
            # testing only cnn
            sess.run(tf.global_variables_initializer())
            imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            probs = model.test_cnn(imgs)
            model.load_cnn(sess, FLAGS.cnn_model_file)

            img1 = imread(FLAGS.image_file, mode='RGB')
            img1 = imresize(img1, (224, 224))

            prob = sess.run(probs, feed_dict={imgs: [img1]})[0]
            preds = (np.argsort(prob)[::-1])[0:5]
            for p in preds:
                print(class_names[p], prob[p])

        else:
            # testing phase
            test_data= prepare_test_data(config)
            model.load(sess, FLAGS.model_file)
            #tf.get_default_graph().finalize()
            model.test(sess, test_data)

if __name__ == '__main__':
    tf.app.run()