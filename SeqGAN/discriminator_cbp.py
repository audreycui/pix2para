import tensorflow as tf
from baseModel import BaseModel
from tensorflow.contrib import slim
import numpy as np
from utils.compact_bilinear_pooling import compact_bilinear_pooling_layer as cbp

#from Shaofan Lai's tensorflow implementation of SeqGAN (Yu et. al, 2017)
#https://github.com/Shaofanl/SeqGAN-Tensorflow
#SeqGAN: https://arxiv.org/abs/1609.05473

#modified discriminator to condition on image features to make truth predictions
#image features and text are combined using concatenation
#TODO: compact bilinear pooling has been shown to be an effective way to combine multimodal features, but currently does not work bc of windows OS incompatibility
    #as described in https://github.com/richardaecn/cvpr18-caption-eval (Cui et. al, 2018)
class Discriminator_CBP(object):
    def __init__(self, parent, config):
        
        self.parent = parent
        self.config = config
        batch_size = config.batch_size
        seq_len = config.max_caption_length
        emb_size = config.dim_embedding
        hidden_size = config.D_hidden_size
        filter_sizes = config.d_filter_sizes
        num_filters = config.d_num_filters
        l2_reg_lambda = 0
        combine_type = config.combine_type
        #print("dis batch size: " + str(batch_size))
        # Placeholders for input, output and dropout
        input_x = tf.placeholder(
            tf.int32, [batch_size, seq_len], name="input_x")
        input_y = tf.placeholder(
            tf.int32, [batch_size], name="input_y")
        dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        conv_features = tf.placeholder(
                tf.float32, [batch_size, config.image_feat_dim], 
                name="conv_features") 

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        #lstm for the img_feats
        

        with tf.variable_scope('discriminator'):
            # Embedding layer
            emb = tf.Variable(
                tf.random_uniform([config.vocabulary_size, emb_size], -1.0, 1.0),
                name="W")
            emb_x = tf.nn.embedding_lookup(emb, input_x)
            #emb_x_expand = tf.expand_dims(emb_x, -1)
            #print(emb_x_expand, emb_x)

            img_feat = tf.layers.dense(inputs=conv_features, units=hidden_size, activation=None)
            img_feat = tf.expand_dims(img_feat, 1)
            
            img_feat = tf.tile(img_feat, [1, seq_len, 1])
            print('emb_x ' + str(emb_x.shape))
            print('img_feat ' + str(img_feat.shape))
            if combine_type == 'concat':
                conv_input = tf.concat([emb_x, img_feat], axis = 2)
                conv_input = tf.expand_dims(conv_input, axis = -1)

            #TODO figure out how to call c libraries for bilinpool
            elif combine_type == 'bilinpool': 
                out_dim = hidden_size*2
                #emb_x = tf.expand_dims(emb_x, axis = -1)
                #img_feat = tf.expand_dims(img_feat, axis = -1)
                conv_input = cbp(emb_x, img_feat, out_dim)
                conv_input = tf.expand_dims(conv_input, axis = -1)
           
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, pair in enumerate(zip(filter_sizes, num_filters)):
                filter_size, num_filter = pair
                #print (filter_size, num_filter, pair)
                with tf.name_scope("conv-maxpool-%d" % i):
                    # Convolution Layer
                    filter_shape = [filter_size, emb_size*2, 1, num_filter] #emb*2 for concatenated img and text feats
                    W = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]),
                                    name="b")
                    conv = tf.nn.conv2d(
                        conv_input,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, seq_len-filter_size+1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            
            #print("pooled_outputs" + str(pooled_outputs))
            #shape of pooled outputs: [batch_size, 2, 1, total filters]

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            #print('num_filters_total' + str(num_filters_total))
            # Add highway
            #print("h_pool_flat" + str(h_pool_flat.shape))
            with tf.name_scope("highway"):
                h_highway = highway(h_pool_flat,
                                    h_pool_flat.get_shape()[1], 
                                    config.highway_layers, 0)
            # Add dropout
            h_highway = h_pool_flat

            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, dropout_keep_prob) 
            #print('h_drop: ' + str(h_drop.shape))
            
            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal(
                    [num_filters_total, 2], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
                truth_prob = tf.nn.softmax(scores, -1)[:, 1]
                predictions = tf.argmax(scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            #print("SCORES SHAPE: " + str(scores.shape))
            with tf.name_scope("loss"):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=scores, labels=input_y)
                loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            with tf.name_scope("accuracy"):
                acc = tf.reduce_mean(tf.to_float(
                    tf.equal(tf.cast(predictions, tf.int32), input_y)))

        with tf.variable_scope("dis_adam", reuse = tf.AUTO_REUSE): #added var scope for optimizer
            d_opt = tf.train.AdamOptimizer(1e-4)
            train_op = tf.contrib.slim.learning.create_train_op(
                loss, d_opt, aggregation_method=2)

        self.d_summary = tf.summary.merge([
            tf.summary.scalar("d_loss", loss),
            tf.summary.scalar("d_acc", acc),
        ])

        self.train_op = train_op
        self.predictions = predictions
        self.truth_prob = truth_prob

     
        self.input_x = input_x
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob
        self.conv_features = conv_features

        print ("Discriminator is built.")

    def predict(self, sess, input_x, conv_features):
        feed_dict = {self.input_x: input_x, self.conv_features: conv_features,
                     self.dropout_keep_prob: 1.0}
        return sess.run(self.predictions, feed_dict=feed_dict)

    def get_truth_prob(self, sess, input_x, conv_features):
        feed_dict = {self.input_x: input_x, self.conv_features: conv_features,
                     self.dropout_keep_prob: 1.0}
        return sess.run(self.truth_prob, feed_dict=feed_dict)

    def train(self, sess, input_x, input_y, conv_features, dropout_keep_prob=0.75):
        #input_x: input sentence tokens
        #input_y: labels
        feed_dict = {self.input_x: input_x,
                     self.input_y: input_y,
                     self.conv_features: conv_features, 
                     self.dropout_keep_prob: dropout_keep_prob}
        _, summary = sess.run([self.train_op, self.d_summary],
                              feed_dict=feed_dict)
        return summary


def highway(input, size, num_layers=1, bias=-2.0,
            f=tf.nn.relu, scope='Highway'):
    """
        Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1-t) * y
        where g is nonlinearity, t is transform gate, and (1-t) is carry gate.
    """
    #print("enter highway")
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE): #added auto reuse
        size = int(size)
        #output = tf.Variable(input, tf.float32)
        
        for idx in range(num_layers):
            #g = f(slim.fully_connected(
            g = tf.nn.relu(slim.fully_connected(
                input, size,
                scope= 'highway_lin_%d' % idx,
                activation_fn=None))

            t = tf.sigmoid(slim.fully_connected(
                input, size,
                scope='highway_gate_%d' % idx,
                activation_fn=None) + bias)

            output = t * g + (1. - t) * input
            input = output
        
    #print(output)
    return output
