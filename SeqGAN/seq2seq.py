import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

from baseModel import BaseModel

from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import tensorflow.contrib.seq2seq.python.ops.attention_wrapper
import tensorflow.contrib.seq2seq.python.ops.beam_search_decoder

#code base: https://github.com/google/seq2seq
'''
Added: 
#attention mechanism
#changed encoder lstm from unidirectional to bidirectional
#multilayered decoder
#conditions on image features
'''
class BasicS2SModel(object):
    def __init__(self, parent, config): #replace build

        self.parent = parent #'parent' is seqgan, use parent to call seqgan's methods
        self.config = config

        self.is_train = parent.is_train
        #print("is_train: " + str(self.is_train))
        #assert mode in ['train', 'inference']
        #self.is_pretrain = True
        self.start_token = config._START_
        self.end_token = config._END_
        #self.train_phase = parent.is_train

        self.cell_name = 'gru'
        self.dim_size = config.dim_embedding
        self.vocab_size = config.vocabulary_size + len(config.ctrl_symbols)
        self.num_layers = config.num_decode_layers
        self.keep_prob_config = 1.0 - config.lstm_drop_rate
        self.atten_size = config.atten_size
        self.max_gradient_norm = config.max_gradient_norm
        self.max_source_len = config.image_feat_dim
        self.max_target_len = config.max_caption_length
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        
        # decoder
        self.max_inference_length = config.max_caption_length
        
        # beam search
        self.beam_size = config.beam_size
        self.beam_search = config.use_beam_search
        
        # learning
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        
        # if we use beam search decoder, we need to specify the batch size and max source len
        
        if config.use_beam_search:
            self.batch_size = config.batch_size
            self.source_tokens = tf.placeholder(tf.int32, shape=[self.batch_size, config.image_feat_dim])
            self.source_length = tf.placeholder(tf.int32, shape=[self.batch_size,])
        else:
            self.source_tokens = tf.placeholder(tf.int32,shape=[None,None])
            self.source_length = tf.placeholder(tf.int32,shape=[None,])
        #print("source tokens " + str(self.source_tokens.shape))
        if self.is_train:
            self.target_tokens = tf.placeholder(tf.int32, shape=[None, None])
            self.target_length = tf.placeholder(tf.int32, shape=[None,])
         
        with tf.variable_scope("S2S",initializer = tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            #self.setup_encoder()
            self.setup_bidirection_encoder()
            self.setup_attention_decoder()
            #self.setup_rewards()
                
        if self.is_train:
            opt = tf.train.AdamOptimizer(self.learning_rate)
            params = tf.trainable_variables()
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.gradient_norm = tf.global_norm(gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    
    def setup_embeddings(self):
        with tf.variable_scope("Embeddings"):
            with tf.device('/cpu:0'):
                self.enc_emd = tf.get_variable("encode_embedding", [self.vocab_size, self.dim_size])
                self.dec_emd = tf.get_variable("decode_embedding", [self.vocab_size, self.dim_size])
                self.encoder_inputs = tf.nn.embedding_lookup(self.enc_emd, self.source_tokens)
                if self.is_train:
                    self.decoder_inputs = tf.nn.embedding_lookup(self.dec_emd, self.target_tokens)
    
    def setup_encoder(self):
        cell = multi_rnn_cell(self.cell_name,self.dim_size, self.num_layers, self.is_train,self.keep_prob_config)
        outputs,state = tf.nn.dynamic_rnn(cell,inputs=self.encoder_inputs,sequence_length=self.source_length,dtype=tf.float32)
        self.encode_output = outputs
        self.encode_state = state
        # using the state of last layer of rnn as initial state
        self.decode_initial_state = self.encode_state[-1]
        
    def setup_bidirection_encoder(self):
        fw_cell = single_rnn_cell('gru',self.dim_size, train_phase=self.is_train, keep_prob=self.keep_prob_config)
        bw_cell = single_rnn_cell('gru',self.dim_size, train_phase=self.is_train, keep_prob=self.keep_prob_config)
        
        with tf.variable_scope("Encoder"):
            outputs,states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = fw_cell,
                cell_bw = bw_cell,
                dtype = tf.float32,
                sequence_length = self.source_length,
                inputs = self.encoder_inputs
                )
            outputs_concat = tf.concat(outputs, 2)
        self.encode_output = outputs_concat
        self.encode_state = states
        
        # use Dense layer to convert bi-direction state to decoder inital state
        convert_layer = Dense(self.dim_size,dtype=tf.float32,name="bi_convert")
        self.decode_initial_state = convert_layer(tf.concat(self.encode_state,axis=1))
        
    def setup_training_decoder_layer(self):
        max_dec_len = tf.reduce_max(self.target_length, name='max_dec_len')
        training_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs,self.target_length,name="training_helper")
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.dec_cell,
            helper = training_helper,
            initial_state = self.initial_state,
            output_layer = self.output_layer
        )
        train_dec_outputs, train_dec_last_state,_ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_dec_len)
        
        # logits: [batch_size x max_dec_len x vocab_size]
        logits = tf.identity(train_dec_outputs.rnn_output, name='logits')
        self.output_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-20, 1.0))
        # targets: [batch_size x max_dec_len x vocab_size]
        targets = tf.slice(self.target_tokens, [0, 0], [-1, max_dec_len], 'targets')

        masks = tf.sequence_mask(self.target_length,max_dec_len,dtype=tf.float32,name="mask")
        self.losses = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=targets,weights=masks,name="losses")
        self.loss_summary = tf.summary.scalar(
                "g_pretrain_loss", self.losses)
           
        # prediction sample for validation
        self.predictions = tf.identity(train_dec_outputs.sample_id, name='valid_preds')
        self.predictions = tf.squeeze(self.predictions)
        
        self.setup_rewards() #rewards for adversarial training
    
    def setup_inference_decoder_layer(self):

        #print("INFERENCE")
        start_tokens = tf.tile(tf.constant([self.start_token],dtype=tf.int32),[self.batch_size])
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self.dec_emd,
            start_tokens=start_tokens,
            end_token=self.end_token)
        
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.dec_cell,
            helper = inference_helper, 
            initial_state=self.initial_state,
            output_layer=self.output_layer)
        
        infer_dec_outputs, infer_dec_last_state,_ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.max_inference_length)
        # [batch_size x dec_sentence_length], tf.int32

        #logits = tf.identity(infer_dec_outputs.rnn_output, name='infer_logits')

        #self.output_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-20, 1.0))  # B,Vocab_size # add 1e-8 to prevent log(0)

        
        self.predictions = tf.identity(infer_dec_outputs.sample_id, name='predictions')
        print("predictions shape " + str((self.predictions).shape))

        #self.predictions = tf.squeeze(self.predictions)
        #print(self.predictions)

    def setup_beam_search_decoder_layer(self):
        start_tokens = tf.tile(tf.constant([self.start_token],dtype=tf.int32),[self.batch_size])
        bsd = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.dec_cell,
                    embedding=self.dec_emd,
                    start_tokens= start_tokens,
                    end_token=self.end_token,
                    initial_state=self.initial_state,
                    beam_width=self.beam_size,
                    output_layer=self.output_layer,
                    length_penalty_weight=0.0)
        # final_outputs are instances of FinalBeamSearchDecoderOutput
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            bsd, 
            output_time_major=False,
           # impute_finished=True,
            maximum_iterations=self.max_inference_length
        )
        beam_predictions = final_outputs.predicted_ids
        self.beam_predictions = tf.transpose(beam_predictions,perm=[0,2,1])
        self.beam_prob = final_outputs.beam_search_decoder_output.scores
        self.beam_ids = final_outputs.beam_search_decoder_output.predicted_ids
        
    def setup_attention_decoder(self):
        #dec_cell = multi_rnn_cell('gru',self.dim_size,num_layers=self.num_layers, train_phase = self.train_phase, keep_prob=self.keep_prob_config)
        dec_cell = [single_rnn_cell(self.cell_name,self.dim_size,self.is_train,self.keep_prob_config) for i in range(self.num_layers)]
        if self.beam_search:
            memory = tf.contrib.seq2seq.tile_batch(self.encode_output,multiplier = self.beam_size)
            memory_sequence_length = tf.contrib.seq2seq.tile_batch(self.source_length,multiplier = self.beam_size)
        else:
            memory = self.encode_output
            memory_sequence_length = self.source_length
            
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(
            num_units = self.atten_size,
            memory = memory,
            memory_sequence_length = memory_sequence_length,
            name = "BahdanauAttention"
        )
        dec_cell[0] = tf.contrib.seq2seq.AttentionWrapper(
            cell=dec_cell[0],
            attention_mechanism=attn_mech,
            attention_layer_size=self.atten_size)
        
        if self.beam_search:
            tile_state = tf.contrib.seq2seq.tile_batch(self.decode_initial_state,self.beam_size)
            initial_state = [tile_state for i in range(self.num_layers)]
            cell_state = dec_cell[0].zero_state(dtype=tf.float32,batch_size=self.batch_size*self.beam_size)
            initial_state[0] = cell_state.clone(cell_state=initial_state[0])
            self.initial_state = tuple(initial_state)
        else:
            # we use dynamic batch size
            #self.batch_size = tf.shape(self.encoder_inputs)[0]     
            initial_state = [self.decode_initial_state for i in range(self.num_layers)]
            cell_state = dec_cell[0].zero_state(dtype=tf.float32, batch_size = self.batch_size)
            initial_state[0] = cell_state.clone(cell_state=initial_state[0])
            self.initial_state = tuple(initial_state)
            
        #print(self.initial_state)
        self.dec_cell = tf.contrib.rnn.MultiRNNCell(dec_cell)
        print("dec_cell " + str(dec_cell))
        self.output_layer = Dense(self.vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
        if self.is_train:
            self.setup_training_decoder_layer()
        else:
            if self.beam_search:
                self.setup_beam_search_decoder_layer()
            else:
                self.setup_inference_decoder_layer()

    def setup_rewards(self):

        self.rewards = tf.placeholder(tf.float32,
                             shape=[self.batch_size,],
                             name="rewards")
        g_seq = self.predictions
        g_prob = self.output_probs

        #debugging shape issues
        
        #print("shape of g_prob: " + str(np.shape(g_prob)))
        #print("shape of g_seq: " + str(np.shape(g_seq)))

        one_hot = tf.one_hot(g_seq, self.vocab_size) 
        #print("shape of one hot: " + str(one_hot.get_shape()))
        reduced_sum = tf.reduce_sum(one_hot * tf.log(tf.clip_by_value(g_prob, 1e-20, 1.0)), -1)
        #print("shape of reduce sum: " + str(reduced_sum.get_shape()))
        #print("shape of rewards: " + str(self.rewards.get_shape()))
        rewards_expanded = tf.expand_dims(self.rewards, axis = 1)
        g_loss = -tf.reduce_mean( 
            tf.reduce_sum(tf.one_hot(g_seq, self.vocab_size) * tf.log(tf.clip_by_value(g_prob, 1e-20, 1.0)), -1) * rewards_expanded
        )

        with tf.variable_scope("rl_adam", reuse = tf.AUTO_REUSE): #added var scope for optimizer
            g_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
            g_op = slim.learning.create_train_op(
                g_loss, g_optimizer, clip_gradient_norm=5.0)
        g_summary = tf.summary.merge([
            tf.summary.scalar("g_loss", g_loss),
            tf.summary.scalar("g_reward", tf.reduce_mean(self.rewards))
        ])
 
        #self.conv_features = conv_features
        #self.rewards = rewards
        self.g_op = g_op
        self.g_summary = g_summary
        if self.is_train:
            self.given_tokens = self.target_tokens
            self.image_summary = tf.summary.merge([
                tf.summary.image(
                    "real_samples",
                    tf.expand_dims(tf.one_hot(self.given_tokens, self.vocab_size), -1)
                ),
                tf.summary.image(
                    "fake_samples",
                    tf.expand_dims(tf.one_hot(g_seq[0], self.vocab_size), -1)
                ),
            ])

        self.predictions = g_seq
        self.pred_probs = g_prob
    
    def train_one_step(self,sess,encode_input,encode_len,decode_input,decode_len):
        feed_dict = {}
        feed_dict[self.source_tokens] = encode_input
        feed_dict[self.source_length] = encode_len
        feed_dict[self.target_tokens] = decode_input
        feed_dict[self.target_length] = decode_len

        #print ("encode_input shape " + str(np.array(encode_input).shape))
        #print ("encode_len shape " + str(np.array(encode_len).shape))
        #print ("decode_input shape " + str(np.array(decode_input).shape))
        #print ("decode_len shape " + str(np.array(decode_len).shape))
        
        valid_predictions,loss,_,loss_summary = sess.run([self.predictions,self.losses,self.updates,self.loss_summary],
                                                        feed_dict=feed_dict)
        return valid_predictions,loss, loss_summary

    def train_one_step_rewards(self,sess,encode_input,encode_len,decode_input,decode_len, rewards):
        feed_dict = {}
        feed_dict[self.source_tokens] = encode_input
        feed_dict[self.source_length] = encode_len
        feed_dict[self.target_tokens] = decode_input
        feed_dict[self.target_length] = decode_len
        feed_dict[self.rewards] = rewards
        
        #print ("encode_input shape " + str(np.array(encode_input).shape))
        #print ("encode_len shape " + str(np.array(encode_len).shape))
        #print ("decode_input shape " + str(np.array(decode_input).shape))
        #print ("decode_len shape " + str(np.array(decode_len).shape))
        
        valid_predictions,loss,_, loss_summary = sess.run([self.predictions,self.losses,self.updates, self.g_summary],feed_dict=feed_dict)
        return valid_predictions,loss, loss_summary
    
    def pretrain(self, sess, sentences, conv_features, dec_sentence_lengths):
        #print ("Training the model for one batch..")
        self.is_train = True
        #self.is_pretrain = True
        enc_feat_lengths = np.array([self.config.image_feat_dim] * self.config.batch_size)

        batch_preds, batch_loss, summary = self.train_one_step(sess,conv_features,enc_feat_lengths,sentences,dec_sentence_lengths)
        #epoch_loss += batch_loss
        return summary, batch_preds, batch_loss
        #return loss_history

    def generate(self, sess, given_tokens, conv_features, dec_sentence_lengths):
        #print("Generating tokens...")
        #self.is_train = False
        #self.is_pretrain = False
        enc_feat_lengths = np.array([self.config.image_feat_dim] * self.config.batch_size)

        feed_dict = {self.target_tokens: given_tokens, 
                     self.target_length: dec_sentence_lengths,
                     self.source_tokens: conv_features, 
                     self.source_length: enc_feat_lengths}

        predictions = sess.run([self.predictions], feed_dict=feed_dict)
        return predictions


    def train(self, sess, given_tokens, conv_features, rewards):

        self.is_train = True
        #self.is_pretrain = False
        #input_batch_tokens = []
        #target_batch_tokens = []
        #enc_feat_lengths = []
        dec_sentence_lengths = []

        enc_feat_lengths = np.array([self.config.image_feat_dim] * self.config.batch_size)

        for target_sent in given_tokens:
            #TODO end token for generated samples
            try: 
                dec_sentence_lengths.append(target_sent.tolist().index(self.config._END_)+1)
            except Exception: 
                dec_sentence_lengths.append(len(target_sent.tolist()))
        dec_sentence_lengths = np.array(dec_sentence_lengths)
        #print(enc_feat_lengths)
        #print(dec_sentence_lengths)
        batch_preds, _, summary = self.train_one_step_rewards(sess,conv_features,enc_feat_lengths,given_tokens,dec_sentence_lengths, rewards)

        return summary, batch_preds
    
    def save_model(self,sess,checkpoint_dir):
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess,checkpoint_dir + "model.ckpt",global_step=self.global_step)
        
    def restore_model(self,sess,checkpoint_dir):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))

    def get_reward(self, sess, given_tokens, conv_features, rollout_num, discriminator):
        #print("get_REWARD...")
        #batch_size = tf.shape(self.conv_features)[0]
        batch_size = self.batch_size
        #print("BATCH SIZE: " + str(batch_size))
        seq_len = self.max_inference_length
        rewards = np.zeros((batch_size))
        for i in range(rollout_num):
            # Markov Chain Sample
            mc_sample = self.rollout(sess, given_tokens, conv_features)
            mc_sample = np.squeeze(mc_sample)
            mc_sample = self.pad_fake_samples(mc_sample)

            truth_prob = discriminator.get_truth_prob(sess, mc_sample, conv_features)
            #print("truth prob  " + str(truth_prob))
            rewards += truth_prob
        rewards /= rollout_num
        return rewards
    
    def rollout(self, sess, given_tokens, conv_features, with_probs=False):
        enc_feat_lengths = []
        dec_sentence_lengths = []

        enc_feat_lengths = np.array([self.config.image_feat_dim] * self.config.batch_size)
        #print(enc_feat_lengths)
        for target_sent in given_tokens:
            #print(target_tokens)
            #print("target length:" + str(target_sent.tolist().index(self.config._END_)+1))
            dec_sentence_lengths.append(target_sent.tolist().index(self.config._END_)+1)
        dec_sentence_lengths = np.array(dec_sentence_lengths)
        feed_dict = {self.target_tokens: given_tokens, 
                     self.target_length: dec_sentence_lengths,
                     self.source_tokens: conv_features, 
                     self.source_length: enc_feat_lengths}
        if with_probs:
            #output_tensors = [self.output_ids[keep_steps],
            #                  self.output_probs[keep_steps]]
            return sess.run([self.predictions, self.output_probs], feed_dict=feed_dict)
        else:
            #output_tensors = self.output_ids[keep_steps]
            return sess.run([self.predictions], feed_dict=feed_dict)

    def eval(self, sess, conv_features):
        enc_feat_lengths = []
        for input_feat in conv_features:
            enc_feat_lengths.append(len(input_feat))

        feed_dict = {self.source_tokens: conv_features, self.source_length: enc_feat_lengths}

        if self.config.use_beam_search:
            predictions, probs, ids = sess.run([self.beam_predictions,self.beam_prob,self.beam_ids],feed_dict=feed_dict)
            return predictions
        else:
            predictions = sess.run([self.predictions], feed_dict=feed_dict)
            return predictions

    def pad_fake_samples(self, fake_samples):
        fake_samples_list = []
        for i in range (len(fake_samples)):
            temp = fake_samples[i].tolist()
            len_list = len(fake_samples[i])
            for j in range(len_list, self.max_target_len):
                temp.append(1)
            fake_samples_list.append(temp)
        return np.array(fake_samples_list)
    


def single_rnn_cell(cell_name,dim_size, train_phase = True, keep_prob = 0.75):
    if cell_name == "gru":
        cell = tf.contrib.rnn.GRUCell(dim_size)
    elif cell_name == "lstm":
        cell = tf.contrib.rnn.LSTMCell(dim_size)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(dim_size)
    if train_phase and keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(
              cell=cell,
              input_keep_prob=keep_prob,
              output_keep_prob=keep_prob)
    return cell
