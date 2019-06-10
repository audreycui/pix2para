import numpy as np
import pandas as pd
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model

from utils.misc import  ImageLoader
import config
from sklearn.utils import shuffle
'''
Added: 
#added image features in each batch
#get_imagefeatures_vgg19 function to load or compute image features (VGG19 feature map)
#DataTestLoader class (for testing phase)
#DataValLoader class (for validation phase)
#create_shuffled_batch function in DataLoader: creates batches with random order
'''
class DataLoader():
    def __init__(self, config, batch_size, seq_length, end_token=0):
        self.config = config
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token
        self.image_batch = None
        self.feature_batch = None
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        net = VGG19(weights='imagenet')
        self.trained_model = Model(input= net.input, output= net.get_layer('fc2').output)

    def get_imagefeatures_vgg19(self, image_files, feature_files):
        #print("to extract features...")
        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, feature_files, self.batch_size) #extract image features using vgg19

    def next_batch(self):
        seq = self.sequence_batch[self.pointer]
        imgs = None
        features = None

        if self.image_batch is not None:
            imgs = self.image_batch[self.pointer]
            feat_file = self.feature_batch[self.pointer]
            conv = np.array(self.get_imagefeatures_vgg19(imgs, feat_file))
        else:
            print("no image files")

        self.pointer = (self.pointer + 1) % self.num_batch

        return seq, imgs, feat_file, conv

    def reset_pointer(self):
        self.pointer = 0

    def create_shuffled_batches(self, with_image=True):

        self.pointer = 0
        config = self.config

        data = np.load(config.temp_data_file).item()
        word_idxs = data['word_idxs']
        sent_lens = data['sentence_len']
        print("len word_idxs: " + str(len(word_idxs)))

        self.num_batch = int(len(word_idxs) / self.batch_size)
        print("num batch " + str(self.num_batch))
        print('batch_size' + str(self.batch_size))
        print(self.num_batch * self.batch_size)
        word_idxs = word_idxs[:self.num_batch * self.batch_size]
        
        #self.pointer = 0

        if with_image:
            with open(config.temp_image_file) as ifile:
                image_files = ifile.read().splitlines()
            with open(config.temp_feature_file) as ffile:
                feature_files = ffile.read().splitlines()
            

            image_files = image_files[:self.num_batch * self.batch_size]          
            feature_files = feature_files[:self.num_batch * self.batch_size]
            
            print("len image files: " + str(len(image_files)))
            print("len feature files: " + str(len(feature_files)))
            print("len word_idxs: " + str(len(word_idxs)))

            word_idxs, feature_files, image_files = shuffle(
                word_idxs, feature_files, image_files)

            self.sequence_batch = np.array(np.split(word_idxs, self.num_batch, 0))
            self.image_batch = np.array(np.split(np.array(image_files), self.num_batch, 0))
            self.feature_batch = np.array(np.split(np.array(feature_files), self.num_batch, 0))
            
        else:
            image_files = None
            feature_files = None
            word_idxs = shuffle(word_idxs)
            self.sequence_batch = np.split(word_idxs, self.num_batch, 0)

        print('shape of sequence_batch:' + str(self.sequence_batch.shape))
        print('shape of image:' + str(self.image_batch.shape))
        print('shape of features:' + str(self.feature_batch.shape))

    def get_sample_features(self):
        data = np.load(config.temp_sample_image_file).item()
        imgs = data['images']
        features = data['features']

        return imgs, features

    def reset_image_pointer(self):
        self.image_pointer = 0


class DisDataloader():
    def __init__(self, config, batch_size, seq_length):
        self.config = config
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length

        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        net = VGG19(weights='imagenet')
        self.trained_model = Model(input= net.input, output= net.get_layer('fc2').output)

    def get_imagefeatures_vgg19(self, image_files, feature_files):
        #print("to extract features...")
        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, feature_files, self.batch_size)

    def load_train_data(self, with_image):
        # Load data
        #pos: oracle, neg: generated samples
        data = np.load(self.config.temp_generate_file).item()
        #data = {'feature_files': feature_files, 'real_samples': real_samples, 'generated_samples': generated_samples}
        positive_examples = data['real_samples']
        negative_examples = data['generated_samples']
        feature_files = data['feature_files']
        image_files = data['image_files']

        
        if with_image: #same order as postive and negative examples
            feature_files = np.concatenate([feature_files, feature_files], 0)
            image_files = np.concatenate([image_files, image_files], 0)
            
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)
        
        # Split batches
       
        #self.sentences = np.array(positive_examples + negative_examples)
        self.sentences = np.concatenate([positive_examples, negative_examples], 0)
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        #debug = self.labels
        if with_image:
           
            feature_files = feature_files[:self.num_batch * self.batch_size]
            image_files = image_files[:self.num_batch * self.batch_size]
            self.labels, self.sentences, feature_files, image_files = shuffle(
                self.labels, self.sentences, feature_files, image_files)

            self.feature_batch = np.split(np.array(feature_files), self.num_batch, 0)
            self.image_batch = np.split(np.array(image_files), self.num_batch, 0)
        else:
            self.labels, self.sentences = shuffle(self.labels, self.sentences)

        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def print_sample(self, array):
        return
        for i in range(10):
            print(str(array[i]))

    def next_batch(self):
        sent = self.sentences_batches[self.pointer]
        lab = self.labels_batches[self.pointer]
        imgs = None
        features = None
        if self.image_batch:
            imgs = self.image_batch[self.pointer]
            feature_files = self.feature_batch[self.pointer]
            features = self.get_imagefeatures_vgg19(imgs, feature_files)
        else:
            print("no image files")

        self.pointer = (self.pointer + 1) % self.num_batch
        return sent, lab, features

    def reset_pointer(self):
        self.pointer = 0

class DataEvalLoader():
    def __init__(self, config, batch_size, end_token=0):
        self.config = config
        self.batch_size = batch_size
        self.end_token = end_token
        self.image_batch = None
        self.feature_batch = None
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        net = VGG19(weights='imagenet')
        self.trained_model = Model(input= net.input, output= net.get_layer('fc2').output)

    def get_imagefeatures_vgg19(self, image_files, feature_files):
        #print("to extract features...")
        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, feature_files, self.batch_size) #extract image features using vgg19

    def next_batch(self):
        imgs = None
        features = None

        if self.image_batch is not None:
            imgs = self.image_batch[self.pointer]
            feat_file = self.feature_batch[self.pointer]
            conv = np.array(self.get_imagefeatures_vgg19(imgs, feat_file))
        else:
            print("no image files")


        self.pointer = (self.pointer + 1) % self.num_batch

        return imgs, feat_file, conv

    def reset_pointer(self):
        self.pointer = 0

    def create_batches(self, with_image=True):

        self.pointer = 0
        config = self.config

        if with_image:
            data = pd.read_csv(config.eval_temp_file)
            image_files = [] 
            feature_files = []
            for _, img, feat in data.values:
                image_files.append(img)
                feature_files.append(feat) 
            print("len image files: " + str(len(image_files)))
            print("len feature files: " + str(len(feature_files)))
            self.num_batch = int(len(image_files) / self.batch_size)
            print("num batch" + str(self.num_batch))

            image_files = image_files[:self.num_batch * self.batch_size]          
            feature_files = feature_files[:self.num_batch * self.batch_size]
            
            print("len image files: " + str(len(image_files)))
            print("len feature files: " + str(len(feature_files)))

            self.image_batch = np.array(np.split(np.array(image_files), self.num_batch, 0))
            self.feature_batch = np.array(np.split(np.array(feature_files), self.num_batch, 0))
            
        else:
            image_files = None
            feature_files = None

    def get_sample_features(self):
        data = np.load(config.temp_sample_image_file).item()
        imgs = data['images']
        features = data['features']

        return imgs, features

    def reset_image_pointer(self):
        self.image_pointer = 0

class DataTestLoader():
    def __init__(self, config, batch_size, end_token=0):
        self.config = config
        self.batch_size = batch_size
        self.end_token = end_token
        self.image_batch = None
        self.feature_batch = None
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        net = VGG19(weights='imagenet')
        self.trained_model = Model(input= net.input, output= net.get_layer('fc2').output)

    def get_imagefeatures_vgg19(self, image_files, feature_files):
        #print("to extract features...")
        return self.image_loader.extract_features_vgg19(self.trained_model, image_files, feature_files, self.batch_size) #extract image features using vgg19

    def next_batch(self):
        imgs = None
        features = None

        if self.image_batch is not None:
            imgs = self.image_batch[self.pointer]
            feat_file = self.feature_batch[self.pointer]
            conv = np.array(self.get_imagefeatures_vgg19(imgs, feat_file))
        else:
            print("no image files")

        self.pointer = (self.pointer + 1) % self.num_batch

        return imgs, feat_file, conv

    def reset_pointer(self):
        self.pointer = 0

    def create_batches(self, with_image=True):

        self.pointer = 0
        config = self.config

        if with_image:
            data = pd.read_csv(config.test_temp_file)
            image_files = [] 
            feature_files = []
            for _, img, feat in data.values:
                image_files.append(img)
                feature_files.append(feat) 
            #print("len image files: " + str(len(image_files)))
            #print("len feature files: " + str(len(feature_files)))
            self.num_batch = int(len(image_files) / self.batch_size)
            #print("num batch" + str(self.num_batch))

            image_files = image_files[:self.num_batch * self.batch_size]          
            feature_files = feature_files[:self.num_batch * self.batch_size]
            
            #print("len image files: " + str(len(image_files)))
            #print("len feature files: " + str(len(feature_files)))

            self.image_batch = np.array(np.split(np.array(image_files), self.num_batch, 0))
            self.feature_batch = np.array(np.split(np.array(feature_files), self.num_batch, 0))
            
        else:
            image_files = None
            feature_files = None

    def get_sample_features(self):
        data = np.load(config.temp_sample_image_file).item()
        imgs = data['images']
        features = data['features']

        return imgs, features

    def reset_image_pointer(self):
        self.image_pointer = 0
