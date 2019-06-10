import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json

from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

from baseModel import BaseModel
#from discriminator_cbp import Discriminator_CBP
from discriminator import Discriminator
from generator import Generator
from seq2seq import BasicS2SModel
from utils.coco.pycocoevalcap.eval import COCOEvalCap

import os
import shutil

#from Shaofan Lai's tensorflow implementation of SeqGAN (Yu et. al, 2017)
#https://github.com/Shaofanl/SeqGAN-Tensorflow
#SeqGAN: https://arxiv.org/abs/1609.05473

'''
Added: 
#include conv_features in each training function's feed_dict
#Geneartor and BasicS2SModel are interchangeable
'''
class SeqGAN(BaseModel):
    def __init__(self, config):

    
        super().__init__(config)

        #sentences = np.array(sentences)
        self.generator = Generator(self, config)
        #self.generator = BasicS2SModel(self, config)
        #self.discriminator = Discriminator_CBP(self, config)
        self.discriminator = Discriminator(self, config)
        self.log_generation = False

        

    def get_nn(self):
        return self.nn

    #TODO fix this 
    def get_imagefeatures_mxnet(self, image_files): #returns (batch_size, 4096*3)
        images = self.image_loader.load_images_mxnet(image_files)
        return self.image_loader.extract_features_mxnet(self.object_model, self.sentiment_model, self.scene_model, images, self.config.batch_size) #extract image features using vgg19

    
    def get_imagefeatures_vgg19(self, image_files):
        #images = self.image_loader.load_images_vgg19(image_files)

        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, self.config.batch_size) #extract image features using vgg19
    
    def next_batch(self, data, extract=False):
        #print("HERE!!! next batch")
        batch = data.next_batch()
        image_files, sentences, masks, sent_lens = batch
        conv_features = self.get_imagefeatures_vgg19(image_files)
        return sentences, conv_features, sent_lens

    def pad_fake_samples(self, fake_samples):
        fake_samples_list = []
        for i in range (len(fake_samples)):
            temp = fake_samples[i].tolist()
            len_list = len(fake_samples[i])
            for j in range(len_list, self.config.max_caption_length):
                temp.append(1)
            fake_samples_list.append(temp)
        return np.array(fake_samples_list)

    def train(self, sess, train_data):
        '''
        sampler: a function to sample given batch_size

        evaluator: a function to evaluate given the
            generation and the index of epoch
        evaluate: a bool function whether to evaluate
            while training
        '''

        config = self.config
        pretrain_g_epochs = config.pretrain_g_epochs
        #pretrain_g_epochs = 1 #make this 1 for debugging purposes #TODO: when training delete this line
        pretrain_d_epochs = config.pretrain_d_epochs
        #pretrain_d_epochs = 1 #make this 1 for debugging purposes
        #tensorboard_dir = config.log_dir #TODO check what this is
        
        gen = self.generator
        dis = self.discriminator
        batch_size = config.batch_size

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)
        
        fake_samples = []
       
        for epoch in tqdm(list(range(pretrain_g_epochs)), desc='Pretraining Generator'):
        #for epoch in range(1):
            sentences, conv_features, sent_lens = self.next_batch(train_data, True)
            
            summary, fake_samples, loss = gen.pretrain(sess, sentences, conv_features) #changed sampler to next_batch
            #print(np.shape(fake_samples))
            #next_batch consists of images, captions, and masks
            if (epoch%10 == 0):
                for sent, sample in zip(sentences, fake_samples):
                    print("TARGET: " + train_data.vocabulary.get_sentence(sent))
                    
                    print("PREDICTED" + train_data.vocabulary.get_sentence(sample))
                    
                print(">>>>> LOSS " + str(loss))
            writer.add_summary(summary, epoch)
            '''
            if evaluate and evaluator is not None: #TODO add eval
                evaluator(gen.generate(sess), epoch)
            '''
            #TODO evaluator
        if config.debug:
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, self.config.checkpoint_dir+"model.ckpt", global_step=self.generator.global_step)
        else:
            print("saving pretraining")
            self.save(sess)
        

        train_data.reset()

        for epoch in tqdm(list(range(pretrain_d_epochs)), desc='Pretraining Discriminator'):
        #for epoch in range(1):
            sentences, fake_conv_features, sent_lens = self.next_batch(train_data, True)
            fake_samples = gen.generate(sess, sentences, fake_conv_features, sent_lens)
            fake_samples = np.squeeze(fake_samples)

            fake_samples = self.pad_fake_samples(fake_samples)
            #print("fake_samples shape " + str(fake_samples.shape))
            if (epoch%50 == 0):
                print("TARGET: " + train_data.vocabulary.get_sentence(sentences[0]))
                print("PREDICTED: " + train_data.vocabulary.get_sentence(fake_samples[0]))
                print(' ')
            
            real_samples, real_conv_features, sample_lens = self.next_batch(train_data)

           
            #print("real samples shape: " + str(real_samples.shape))
            samples = np.concatenate([fake_samples, real_samples])
            #print("dis samples shape: " + str(samples.shape))
            conv_features = np.concatenate([fake_conv_features, real_conv_features])
            labels = np.concatenate([np.zeros((batch_size,)),
                                     np.ones((batch_size,))])
            for _ in range(3):
                indices = np.random.choice(
                    len(samples), size=(batch_size,), replace=False)
                dis.train(sess, samples[indices], labels[indices], conv_features[indices])
        train_data.reset()
        
        for epoch in tqdm(list(range(config.total_epochs)), desc='Adversarial training'):
        #for epoch in range(1):
            for _ in range(1):
                sentences, conv_features, sent_lens = self.next_batch(train_data, True)
                #real_samples = sentences
                fake_samples = gen.generate(sess, sentences, conv_features, sent_lens)
                fake_samples = np.squeeze(fake_samples) #generator generates fake samples
                fake_samples = self.pad_fake_samples(fake_samples)
                if (epoch%50 == 0):
                    print("TARGET: " + train_data.vocabulary.get_sentence(sentences[0]))
                    print("PREDICTED: " + train_data.vocabulary.get_sentence(fake_samples[0]))
                    print(' ')
            
                rewards = gen.get_reward(sess, sentences, conv_features, config.num_rollout, dis) 

                #debug: changed fake samples to sentences (real samples)
                summary, fake_samples = gen.train(sess, fake_samples, conv_features, rewards) #generate new fake samples and reward
                
                # np.set_printoptions(linewidth=np.inf,
                #                     precision=3)
                # print rewards.mean(0)
            writer.add_summary(summary, epoch)

            for _ in tqdm(list(range(5)), desc='Discriminator batch'):
                #print("conv features shape " + str(np.array(conv_features).shape))
                sentences, fake_conv_features, sent_lens = self.next_batch(train_data, True)

                fake_samples = gen.generate(sess, sentences, fake_conv_features, sent_lens)
                fake_samples = np.squeeze(fake_samples) #generator generates fake samples after being trained
                #real_samples = sentences
                fake_samples = self.pad_fake_samples(fake_samples)
                real_samples, real_conv_features, sample_lens = self.next_batch(train_data, True)
                #TODO pass in image condition for conditional gan
                #print("fake_samples shape " + str(fake_samples.shape))
                #print("real_samples shape " + str(real_samples.shape))
                samples = np.concatenate([fake_samples, real_samples])
                labels = np.concatenate([np.zeros((batch_size,)),
                                         np.ones((batch_size,))])
                conv_features = np.concatenate([fake_conv_features, real_conv_features])

                for _ in range(3):
                    indices = np.random.choice(
                        len(samples), size=(batch_size,), replace=False)
                    summary = dis.train(sess, samples[indices],
                                        labels[indices], conv_features[indices]) #discriminator trains on the fake and real samples
            writer.add_summary(summary, epoch)

            if self.log_generation:
                summary = sess.run(
                    gen.image_summary,
                    feed_dict={gen.given_tokens: real_samples})
                writer.add_summary(summary, epoch)

            '''
            if evaluate and evaluator is not None:
                evaluator(gen.generate(sess), pretrain_g_epochs+epoch)
            '''
            real_samples, conv_features, sample_lens = self.next_batch(train_data, True)
            np.save('generation', gen.generate(sess, real_samples, conv_features, sample_lens))
        if not os.path.exists(config.save_dir):
            try:  
                os.mkdir(config.save_dir)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
            else:  
                print ("Successfully created the directory %s " % path)
        self.save(sess)
        writer.close()
        print("Training complete.")

    def eval(self, sess, eval_data):
        """ Evaluate the model using the COCO val2014 data. """
        print("Evaluating the model ...")
        config = self.config

        results = []
        if not os.path.exists(config.eval_result_dir):
            os.mkdir(config.eval_result_dir)

        #if config.debug:
        self.restore_model(sess)
        
        idx = 0
        eval_epochs = 500
        for k in tqdm(list(range(min(eval_epochs, eval_data.num_batches))), desc='batch'):
        #for k in range(1):
            image_files = eval_data.next_batch()
 
            caption_data = self.generator.eval(sess, conv_features)
            caption_data = np.squeeze(caption_data)
            
            #print('caption data shape ' + str(caption_data.shape))

            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                ## self.predictions will return the indexes of words, we need to find the corresponding word from it.
                word_idxs = caption_data[l]
                ## get_sentence will return a sentence till there is a end delimiter which is '.'
                caption = str(eval_data.vocabulary.get_sentence(word_idxs))
                print(caption)
                results.append({'image_id': int(eval_data.image_ids[idx]),
                                'caption': caption})
                #print(results)
                idx += 1

                # Save the result in an image file, if requested
                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = image_file.split(os.sep)[-1]
                    image_name = os.path.splitext(image_name)[0]
                    img = mpimg.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir,
                                             image_name+'_result.jpg'))

        fp = open(config.eval_result_file, 'w')
        json.dump(results, fp)
        fp.close()

        # Evaluate these captions
        eval_result_coco = eval_data.coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_data.coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def restore_model(self,sess):
        checkpoint_dir = self.config.save_dir
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))

    def save(self,sess):
        checkpoint_dir = self.config.save_dir
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess,checkpoint_dir + "model.ckpt",global_step=self.global_step)
        