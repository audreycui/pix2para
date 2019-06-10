import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

from baseModel import BaseModel

#modifications to seqgan/show and tell code: 
    #added build_vgg19 method using keras
    #added lstm encoder
    #added batch normalization method
    #removed reduce_mean of convnet features
    #changed argmax to monte carlo for sampling next word when training rnn
    #moved build functions (cnn, rnn, optimizer) to constructor
    #added pretrain function
    #added generate function
   
class Generator(object):
    def __init__(self, parent, config): #replace build

        self.parent = parent #'parent' is seqgan, use parent to call seqgan's methods
        self.config = config
        self.is_train = parent.is_train

        print("Building the Generator...")
        #added in config: image_feature_dim = 4096, G_hidden_size = 512

        batch_size = config.batch_size
        seq_len = config.max_caption_length

        # Setup the placeholders
        conv_features = tf.placeholder(
                dtype = tf.float32, 
                shape = [batch_size, config.image_feat_dim], 
                name="conv_features") #added
        """
        self.masks = tf.placeholder(
                dtype = tf.float32,
                shape = [batch_size, seq_len],
                name = "masks")
        """
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
            #tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("image_feat", reuse = tf.AUTO_REUSE):
                # name: "generator/image_feat"
                image_feat_W = tf.get_variable("image_feat_W", [config.image_feat_dim, config.G_hidden_size], tf.float32, random_uniform_init)
                image_feat_b = tf.get_variable("image_feat_b", [config.G_hidden_size], tf.float32, random_uniform_init)
            
            with tf.variable_scope("output"):
                # name: "generator/output"
                output_W = tf.get_variable("output_W", [config.G_hidden_size, config.vocabulary_size], tf.float32, random_uniform_init)
                output_b = tf.get_variable("output_b", [config.vocabulary_size], tf.float32, random_uniform_init)
            
        # Setup the word embedding, we can use pre trained word embeddings later //TODO
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.parent.get_nn().fc_kernel_initializer,
                regularizer = self.parent.get_nn().fc_kernel_regularizer,
                trainable = self.is_train)

        #added LSTM to encode image features
        #setup the lstm encoder 
        with tf.variable_scope("lstm_encoder"):
            encoder = tf.nn.rnn_cell.LSTMCell(config.G_hidden_size, state_is_tuple=True)


        # Setup the LSTM decoder
        with tf.variable_scope("lstm_decoder"):
            decoder = tf.nn.rnn_cell.LSTMCell(
                config.G_hidden_size,
                initializer = self.parent.get_nn().fc_kernel_initializer)
            if self.is_train:
                decoder = tf.nn.rnn_cell.DropoutWrapper(
                    decoder,
                    input_keep_prob = 1.0-config.lstm_drop_rate,
                    output_keep_prob = 1.0-config.lstm_drop_rate,
                    state_keep_prob = 1.0-config.lstm_drop_rate)

        if self.is_train:
            given_tokens = tf.placeholder(
                tf.int32, shape=[batch_size, seq_len], name='given_tokens')

        # Prepare to run
        output_ids = [] #indices of predictions
        output_probs = [] #probabilities of output
     
        num_steps = config.max_caption_length
        #image_emb = tf.reduce_mean(self.conv_feats, axis =1) #removed reduce mean
        
        #============================= encoder ===================================================================
        #input conv_feats dimensions: (batch_size, 4096)
        state = encoder.zero_state(batch_size, tf.float32)
        with tf.variable_scope("image_feat") as scope:
            image_feat = self.batch_norm(conv_features[:,:], mode='test', name='')
        image_feat_emb = tf.matmul(image_feat, image_feat_W) + image_feat_b  # B,H
        lstm_input = image_feat_emb
        with tf.variable_scope("lstm_encoder") as scope:
            _, state = encoder(lstm_input, state)
            encoder_state = state
            self.lstm_c, self.lstm_h = state 

    
        last_word = []

        #============================= decoder ===================================================================
        #initial state of decoder lstm is the final state of encoder lstm
        #^the image feats being fed into the encoder is what makes the generator conditional
        last_state = encoder_state 

        start_token = tf.constant(config.START, tf.int32, [batch_size])
        # Generate the words one by one
        for idx in range(num_steps):
            #first step: start token. otherwise, embed last word
            if idx == 0:
                decoder_input = tf.nn.embedding_lookup(embedding_matrix,
                                                       start_token)
            else:
                with tf.variable_scope("word_embedding"):
                    decoder_input = tf.nn.embedding_lookup(embedding_matrix,
                                                          last_word)
                    #print("decoder_input:"+str(decoder_input))
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                if not idx == 0:
                    tf.get_variable_scope().reuse_variables()

                output, state = decoder(decoder_input, last_state)
                memory, _ = state
                #print("output:"+ str(output))

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                ## Logits is of size vocab
                logits = tf.matmul(output, output_W) + output_b
                log_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-20, 1.0))  # B,Vocab_size # add 1e-8 to prevent log(0)
                output_probs.append(log_probs) #output_prob shape: (20, 32, 5000)
                self.probs = log_probs
                # sample once from the multinomial distribution
                # Montecarlo sampling
                prediction = tf.reshape(tf.multinomial(log_probs, 1), [batch_size])   # 1 means sample once
                #prediction is the index of predicted word
                output_ids.append(prediction)
                
            last_state = state
            if self.is_train:
                ##  During training the input to LSTM is fed by user
                last_word = given_tokens[:, idx]
            else:
                # During testing the input to current time stamp of LSTM is the previous time stamp output.
                last_word = prediction

            tf.get_variable_scope().reuse_variables()

        self.output_ids = output_ids
        self.output_probs = output_probs
    
        if self.is_train:
           

            pretrain_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = tf.reshape(given_tokens, [seq_len, batch_size]), logits = self.output_probs))
            with tf.variable_scope("pretrain_adam", reuse = tf.AUTO_REUSE):
                pretrain_optimizer = tf.train.AdamOptimizer(
                                                           learning_rate = config.learning_rate)
                pretrain_op = slim.learning.create_train_op(
                                                    pretrain_loss, pretrain_optimizer, clip_gradient_norm=5.0)

            self.pretrain_loss = pretrain_loss
            self.pretrain_op = pretrain_op
            self.pretrain_summary = tf.summary.scalar(
                "g_pretrain_loss", pretrain_loss)
           
           
        #reinforcement learning
        rewards = tf.placeholder(tf.float32,
                             shape=[batch_size, num_steps],
                             name="rewards")
       

        g_seq = tf.stack(output_ids, axis=1)
        g_prob = tf.stack(output_probs,axis=1)

        #debugging shape issues
        
        #print("shape of output_ids: " + str(np.shape(output_ids)))
        #print("shape of g_seq: " + str(np.shape(g_seq)))
        one_hot = tf.one_hot(g_seq, config.vocabulary_size) 
        #print("shape of one hot: " + str(one_hot.get_shape()))
        reduced_sum = tf.reduce_sum(one_hot * tf.log(tf.clip_by_value(g_prob, 1e-20, 1.0)), -1)
        #print("shape of reduce sum: " + str(reduced_sum.get_shape()))
        #print("shape of rewards: " + str(rewards.get_shape()))
        
        g_loss = -tf.reduce_mean(
            tf.reduce_sum(tf.one_hot(g_seq, config.vocabulary_size) * tf.log(tf.clip_by_value(g_prob, 1e-20, 1.0)), -1) *
            rewards
        )
        with tf.variable_scope("rl_adam", reuse = tf.AUTO_REUSE): #added var scope for optimizer
            g_optimizer = tf.train.AdamOptimizer(
                learning_rate=config.learning_rate)
            g_op = slim.learning.create_train_op(
                g_loss, g_optimizer, clip_gradient_norm=5.0)
        g_summary = tf.summary.merge([
            tf.summary.scalar("g_loss", g_loss),
            tf.summary.scalar("g_reward", tf.reduce_mean(rewards))
        ])
 
        self.conv_features = conv_features
        self.rewards = rewards
        self.g_op = g_op
        self.g_summary = g_summary
        if self.is_train:
            self.given_tokens = given_tokens
            self.image_summary = tf.summary.merge([
                tf.summary.image(
                    "real_samples",
                    tf.expand_dims(tf.one_hot(self.given_tokens, config.vocabulary_size), -1)
                ),
                tf.summary.image(
                    "fake_samples",
                    tf.expand_dims(tf.one_hot(output_ids[0], config.vocabulary_size), -1)
                ),
            ])

        self.predictions = g_seq
        self.pred_probs = g_prob
        print("Generator built.")

    def generate(self, sess, given_tokens, conv_features):
        #print("Generating tokens...")
        #len for debugging

        feed_dict = {self.given_tokens: given_tokens, #removed images bc already got image features
                     self.conv_features: conv_features}
        return sess.run(self.predictions, feed_dict=feed_dict)
    
    def pretrain(self, sess, sentences, conv_features):
        #print ("Training the model for one batch..")
        #len for debugging

        feed_dict = {self.given_tokens: sentences, 
                     self.conv_features: conv_features}
        loss, summary, predictions, lstm_h = sess.run([self.pretrain_loss, self.pretrain_summary, self.predictions, self.lstm_h],
                              feed_dict=feed_dict)
        print("lstm state:", lstm_h)
        #print ("One run of training data is done")
        return summary, predictions, loss

    def train(self, sess, given_tokens, conv_features, rewards):
        #print("Training the model...")
        feed_dict = {self.conv_features: conv_features,
                     self.given_tokens: given_tokens,
                     self.rewards: rewards}
        _, summary, predictions = sess.run([self.g_op, self.g_summary, self.predictions],
                              feed_dict=feed_dict)
        return summary, predictions
        
    def eval(self, sess, conv_features):
        feed_dict = {self.conv_features: conv_features}
        predictions, probs = sess.run([self.predictions, self.pred_probs], feed_dict=feed_dict)
        return predictions, probs

    #get rewards from discriminator
    def get_reward(self, sess, given_tokens, conv_features, rollout_num, discriminator):
        #print("get_reward...")
        batch_size = self.config.batch_size
        seq_len = self.config.max_caption_length
        rewards = np.zeros((batch_size, seq_len))
        for keep_num in range(1, seq_len):
            for i in range(rollout_num):
                # Markov Chain Sample
                mc_sample = self.rollout(
                                    sess, given_tokens, conv_features, keep_steps=keep_num)
                rewards[:, keep_num] += discriminator.get_truth_prob(sess, mc_sample)
        rewards /= rollout_num
        return rewards

    def rollout(self, sess, given_tokens, conv_features, keep_steps=0, with_probs=False):
        feed_dict = {self.given_tokens: given_tokens,
                     self.conv_features: conv_features}
        if with_probs:
            #output_tensors = [self.output_ids[keep_steps],
            #                  self.output_probs[keep_steps]]
            output_tensors = [self.predictions,
                              self.pred_probs]
        else:
            #output_tensors = self.output_ids[keep_steps]
            output_tensors = self.predictions
        return sess.run(output_tensors, feed_dict=feed_dict)

        #added batch normalization helper method
    def batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)


        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
