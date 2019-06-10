from time import time
import os
import numpy as np
from models.Gan import Gan
from models.leakgan.LeakganDataLoader import DataLoader, DisDataloader, DataTestLoader, DataEvalLoader
from models.leakgan.LeakganDiscriminator import Discriminator
from models.leakgan.LeakganGenerator import Generator
from models.leakgan.LeakganReward import Reward
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleLstm import OracleLstm
from utils.utils import *
from utils.text_process import code_to_text
from utils.text_process import text_precess, text_to_code, process_train_data, process_test_data, process_val_data
from utils.text_process import get_tokenlized, get_word_list, get_dict
from utils.prepare_SPICE import prepare_json
from tqdm import tqdm
import pandas as pd 
import sys
import json
import csv 

'''
Added: 
#test function
#val (validation) function
#save model checkpoint
#restore model checkpoint
#save a context file during generate_samples_gen that records order of shuffled batch 
    #^ensures that real samples can be matched with corresponding fake samples during discriminator training
#include conv_features in each training function's feed_dict
#TODO: batch_norm and in training evaluation metrics currently throw out of memory errors, so both have been removed for now
'''

def pre_train_epoch_gen(sess, trainable_model, data_loader, writer, epoch):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in tqdm(list(range(data_loader.num_batch)), desc="pretraining"):
        sentences, _, _, conv_features = data_loader.next_batch()
        #print("shape" + str(features.shape))
        _, g_loss, _, _, summary = trainable_model.pretrain_step(sess, sentences, conv_features, .8)
        #print(conv_features)
        supervised_g_losses.append(g_loss)
        writer.add_summary(summary, epoch)

    return np.mean(supervised_g_losses)


def generate_samples_gen(sess, trainable_model, data_loader, batch_size, generated_num, 
    output_file=None, context_file = None, get_code=True, train=0, test=False, eval = False):
    # Generate Samples
    generated_samples = []
    real_samples = []
    feature_files = []
    image_files = []
    print("int(generated_num / batch_size)" + str(int(generated_num / batch_size)))
    for i in range(int(generated_num / batch_size)):
        if test:
            imgs, feat_files, conv_features = data_loader.next_batch()
            generated_samples.extend(trainable_model.generate(sess, conv_features, 1.0, 1))
            feature_files.extend(feat_files)
            image_files.extend(imgs)
        else:
            sentences, imgs, feat_files, conv_features = data_loader.next_batch()
            generated_samples.extend(trainable_model.generate(sess, conv_features, 1.0, train))
            real_samples.extend(sentences)
            feature_files.extend(feat_files)
            image_files.extend(imgs)
    
    print("gen samples len: " + str(len(generated_samples)))
    if context_file is not None: 
        data = {'feature_files': feature_files, 'image_files': image_files, 'real_samples': real_samples, 'generated_samples': generated_samples}
        np.save(context_file, data)

    codes = list()
    if output_file is not None:  
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                #print(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)

    if eval is True: 
        return image_files, generated_samples

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    #print(codes)
    return codes


class Leakgan(Gan):
    def __init__(self, config, oracle=None):
        super().__init__(config)
        # you can change parameters, generator here
        #self.vocab_size = 20
        self.emb_dim = config.dim_embedding
        self.hidden_dim = self.emb_dim
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_boolean('restore', False, 'Training or testing a model')
        flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
        flags.DEFINE_integer('length', 70, 'The length of toy data')
        flags.DEFINE_string('model', "", 'Model NAME')
        self.sequence_length = config.max_caption_length #FLAGS.length
        self.filter_size = config.filter_size
        self.num_filters = config.num_filters
        self.l2_reg_lambda = config.l2_reg_lambda
        self.dropout_keep_prob = config.dropout_keep_prob
        self.batch_size = config.batch_size
        self.generate_num = config.generate_num
        self.start_token = config._START_
        self.dis_embedding_dim =  self.emb_dim
        self.goal_size = config.goal_size

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.context_file = config.temp_generate_file
        self.test_file = 'save/test_file.txt'
        self.save_loc = 'save/checkpoints'
        self.global_step = tf.Variable(0, trainable=False)


    def train_discriminator(self):

        generate_samples_gen(self.sess, self.generator, self.gen_data_loader, self.batch_size, self.generate_num, self.generator_file, self.context_file)
        self.dis_data_loader.load_train_data(with_image = True)
        for epoch in range(3):
            #print("training discriminator...")
            x_batch, y_batch, conv_features = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.D_input_x: x_batch,
                self.discriminator.D_input_y: y_batch,
                self.discriminator.conv_features: conv_features
            }
            _, _, summary = self.sess.run([self.discriminator.D_loss, self.discriminator.D_train_op, self.discriminator.D_summary], feed)
            self.writer.add_summary(summary, self.epoch)

            self.generator.update_feature_function(self.discriminator)

    def evaluate(self):
        return #TODO: in-training evaluation metrics throw an out of memory error

    def init_real_training(self, data_loc=None, with_image=True):
        
        self.sequence_length, self.vocab_size, vocabulary = process_train_data(self.config, data_loc, has_image=with_image)
        ##self.sequence_length, self.vocab_size, index_word_dict = text_precess(data_loc, oracle_file=self.config.temp_oracle_file)
        print("sequence length:", self.sequence_length, " vocab size:", self.vocab_size)
        goal_out_size = sum(self.num_filters)

        discriminator = Discriminator(self.config)
        self.set_discriminator(discriminator)

        generator = Generator(self.config, D_model=discriminator)
        self.set_generator(generator)

        # data loader for generator and discriminator
        gen_dataloader = DataLoader(self.config, batch_size=self.batch_size, seq_length=self.sequence_length)
        gen_dataloader.create_shuffled_batches(with_image)
        #gen_dataloader.create_shuffled_batches()

        oracle_dataloader = None
        dis_dataloader = DisDataloader(self.config, batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

        #print("done initializing training")
        return vocabulary

    def init_real_metric(self):
        #from utils.metrics.DocEmbSim import DocEmbSim
        #docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        #self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)
        print("done initializing metric")

    def train_real(self, data_loc=None, with_image=True):
        from utils.text_process import get_tokenlized
        vocabulary = self.init_real_training(data_loc, with_image)
        #self.init_real_metric()

        def get_real_test_file(codes, vocab=vocabulary):
            return
            #with open(self.generator_file, 'r') as file:
            #    codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(vocab.code_to_text(codes))
        
        self.sess.run(tf.global_variables_initializer())
        #self.restore_model(self.sess)
        
        if not os.path.exists(self.config.summary_dir):
            os.mkdir(self.config.summary_dir)
        self.writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

        self.pre_epoch_num = 0
        self.adversarial_epoch_num = 40
        self.log = open('experiment-log-leakgan-real.csv', 'w')
       
        for a in range(0):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1, self.generator.conv_features: np.zeros((self.generator.batch_size, self.generator.image_feat_dim), dtype=np.float32)})
            #print(g)
        #print('start pre-train generator:')
        for epoch in tqdm(list(range(self.pre_epoch_num)), desc='Pretraining generator'):
            start = time()
            loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader, self.writer, self.epoch)
            end = time()
            #print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                codes = generate_samples_gen(self.sess, self.generator, self.gen_data_loader, self.batch_size, self.generate_num, self.generator_file, self.context_file)
                print(vocabulary.code_to_text(codes))
                get_real_test_file(codes)
                self.evaluate()
        self.save_model(self.sess, self.save_loc)

        #print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in tqdm(list(range(self.pre_epoch_num)), desc='Pretraining discriminator'):
      
            self.train_discriminator()
            self.add_epoch()

        self.save_model(self.sess, self.save_loc)

        self.reset_epoch()
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        for epoch in tqdm(list(range(self.adversarial_epoch_num//10)), desc='Adversarial training'):
            for epoch_ in range(10):
                #print('epoch:' + str(epoch) + '--' + str(epoch_))
                start = time()
                for index in range(1):
                    _, _, _, conv_features = self.gen_data_loader.next_batch()
                    samples = self.generator.generate(self.sess, conv_features, 1)
                    rewards = self.reward.get_reward(samples, conv_features)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1.0,
                        self.generator.conv_features: conv_features
                    }
                    _, _, g_loss, w_loss = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, ], feed_dict=feed)
                    print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
                end = time()
                self.add_epoch()
                #print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                    codes = generate_samples_gen(self.sess, self.generator, self.gen_data_loader, self.batch_size, self.generate_num, self.generator_file, self.context_file)
                    print(vocabulary.code_to_text(codes))
                    get_real_test_file(codes)
                    self.evaluate()

                for _ in range(3):
                    self.train_discriminator()
            self.save_model(self.sess, self.save_loc)
            for epoch_ in range(5):
                start = time()
                loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader, self.writer, self.epoch)
                end = time()
                #print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                self.add_epoch()
                if epoch % 5 == 0:
                    codes = generate_samples_gen(self.sess, self.generator, self.gen_data_loader,self.batch_size, self.generate_num,
                                         self.generator_file, self.context_file)
                    print(vocabulary.code_to_text(codes))
                    get_real_test_file(codes)
                    self.evaluate()
            for epoch_ in range(5):
                #print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator()

        self.save_model(self.sess, self.save_loc)
        self.writer.close() 
        
    def test(self, data_loc=None, with_image=True):
        
        goal_out_size = sum(self.num_filters)

        self.sequence_length, self.vocab_size, vocabulary = process_test_data(self.config)

        discriminator = Discriminator(self.config)
        self.set_discriminator(discriminator)

        generator = Generator(self.config, D_model=discriminator)
        self.set_generator(generator)

        # data loader for generator and discriminator
        gen_dataloader = DataEvalLoader(self.config, batch_size=self.batch_size)
        gen_dataloader.create_batches(with_image)
        #gen_dataloader.create_shuffled_batches()

        oracle_dataloader = None
        dis_dataloader = DisDataloader(self.config, batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

        self.restore_model(self.sess)
        #self.sess.run(tf.global_variables_initializer())
        self.context_file = self.config.temp_generate_eval_file
        codes = generate_samples_gen(
            self.sess, self.generator, self.gen_data_loader, self.batch_size, self.batch_size, self.generator_file, test = True)
        samples = vocabulary.code_to_text(codes)
        print(np.array(samples).shape)
        samples = self.remove_padding(samples)
        print(np.array(samples).shape)
        print(samples)

        results_writer = open(self.config.test_result_file, 'w')
        for samp in samples:
            results_writer.write(samp)
        results_writer.close()

    def val(self, data_loc=None, with_image=True):
        
        goal_out_size = sum(self.num_filters)

        self.sequence_length, self.vocab_size, vocabulary = process_val_data(self.config)

        discriminator = Discriminator(self.config)
        self.set_discriminator(discriminator)

        generator = Generator(self.config, D_model=discriminator)
        self.set_generator(generator)

        # data loader for generator and discriminator
        gen_dataloader = DataEvalLoader(self.config, batch_size=self.batch_size)
        gen_dataloader.create_batches(with_image)
        #gen_dataloader.create_shuffled_batches()

        oracle_dataloader = None
        dis_dataloader = DisDataloader(self.config, batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

        self.restore_model(self.sess)
        #self.sess.run(tf.global_variables_initializer())
        #self.context_file = self.config.temp_generate_eval_file
        image_files, codes = generate_samples_gen(
            self.sess, self.generator, self.gen_data_loader, self.batch_size, self.config.num_eval_samples, 
            eval = True, test =True)

        generated_samples = []
        for code in codes:
            #print(code)
            code = vocabulary.code_to_text([code])
            code = self.remove_padding(code)
            generated_samples.append(code)

        np.save(self.config.temp_generate_eval_file, generated_samples)

        ids = []
        for img in image_files:
            #print(img)
            jpg_idx = img.find('.jpg')
            #print(str(jpg_idx))
            ids.append(int(img[45:jpg_idx]))
        np.save(self.config.temp_eval_id, ids)
        prepare_json(self.config)
        
    def remove_padding(self, samples):
        text = samples
        text = str(text)

        text = text[text.find(':')+1:]
        ind1 = text.find('<')
        ind2 = text.find('>')

        while (ind1 >= 0 and ind2 >= 0): 
            text = text[0:ind1] + text[ind2+1:]
            ind1 = text.find('<')
            ind2 = text.find('>')
        ret = text
        return ret

    def save_model(self,sess,checkpoint_dir):
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess,checkpoint_dir + "model.ckpt",global_step=self.global_step)

    def restore_model(self,sess):
        checkpoint_dir = self.config.save_dir
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))

