import datetime
import time

#added config class to make hyperparam tuning and file location changes easier
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):

        #added time stamp to save directories 
        ts = time.time()
        self.st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

        self.debug = False
        self.seed = 17

        self.num_epochs = 200

        self.with_image = True

        # about the model architecture
        self.cnn = 'vgg19'               # changed from vgg16 to vgg19
        self.max_caption_length = 50
        #self.max_caption_length = 100
        self.vocabulary_size = 12000
        #self.vocabulary_size = 10000


        self.hidden_size_global = 256
        self.dim_embedding = self.hidden_size_global
        self.num_lstm_units = self.hidden_size_global
        self.num_initalize_layers = 1 ## Changed from 2 to 1    # 1 or 2
        self.dim_initalize_layer = self.hidden_size_global
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = self.hidden_size_global
        self.num_decode_layers = 3    ## Changed from 2 to 1   # 1 or 2
        self.dim_decode_layer = 1024
        self.image_feat_dim = 4096*1 #for both image features
        self.G_hidden_size = self.hidden_size_global
        self.D_hidden_size = self.hidden_size_global
        self.combine_type = 'concat' #'concat' or 'bilinear pooling'
        self.goal_size = 16
        #added for seq2seq
        self.max_gradient_norm = 5.0
        self.atten_size = 30 #attention size
        self.beam_size = 3
        self.use_beam_search = False


        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.2
        self.lstm_drop_rate = 0.2
        self.attention_loss_factor = 0.01
        self.dropout_keep_prob = 0.75
        # about the optimization

        self.total_epochs = 1500 #added for gan
        self.pretrain_g_epochs=50 #added for gan
        self.pretrain_d_epochs=1000 #added for gan
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        
        self.num_rollout = 16
        self.highway_layers = 5 # added for dis
        self.batch_size = 12
        self.num_generate = self.batch_size
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.learning_rate = 0.001 # for seqgan
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.98
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.l2_reg_lambda = 0.2
        self.temperature = 1
        self.step_size = 4

        # about the saver
        self.save_period = 1000
        #self.save_dir = 'D:/dev/show_and_tell/models_gan_art/'
        self.save_dir = 'D:/test/Texygen/save/model_3/'
        
        self.summary_dir = 'D:/dev/show_and_tell/summary_leakgan' + self.st+'/'
        self.log_dir = 'D:/dev/show_and_tell/logs_summary_leakgan/'
        self.eval_log_dir = 'D:/dev/show_and_tell/elogs/' #TODO eval
        
        self.checkpoint_dir = 'D:/test/checkpoint/'
        # about the training

        # Dataset - COCO
        #base_dir = 'D:/download/COCO'
        #self.ignore_file = base_dir + '/ignore.csv'
        #self.train_caption_file = base_dir + '/train/captions_train2014.json'
        #self.eval_caption_file = base_dir + '/val/captions_val2014.json'
        #self.vocabulary_size = 5000

        # Dataset - art description
        base_dir = 'D:/download/art_desc'
        self.train_caption_file = base_dir + '/train/ann.csv'
        self.eval_caption_file = base_dir + '/val/ann.csv'
        self.caption_file = base_dir + '/train/caption.txt'

        # vocabulary
        self.ctrl_symbols = ['<S>', '<P>', '<E>', '<UNK>']
        self.total_vocabulary_size  = self.vocabulary_size + len(self.ctrl_symbols)

        self.vocabulary_file = base_dir + '/vocabulary.csv'
        #self.START = 0
        self._START_ = 0
        self._PAD_ = 1
        self._END_ = 2
        self._UNK_ = 3

        self.ignore_file = base_dir + '/train/ignore.csv'
        self.ignore_file_eval = base_dir + '/val/ignore.csv'

        self.train_image_dir = base_dir + '/train/images_redo/'
        self.train_feature_dir = base_dir + '/train/images_vgg_redo/'
        self.temp_annotation_file = base_dir + '/train/temp_ann.csv'
        self.temp_data_file = base_dir + '/train/data.npy'
        self.temp_image_file = base_dir + '/train/image_files.npy'
        self.temp_image_shuffle_file =  base_dir + '/train/image_shuffle.npy'
        self.temp_generate_file =  base_dir + '/train/temp_generate.npy'
        self.temp_feature_file = base_dir + '/train/feature_files.txt'
        self.temp_oracle_file = base_dir + '/train/oracle.txt'
        self.temp_generator_file = base_dir + '/train/generator.txt'
        self.temp_sample_image_file = base_dir + '/train/sample_image_files.npy'

  
        self.save_eval_result_as_image = False

        # about the testing
        
        self.test_image_dir = base_dir + '/val/test_images/'
        self.test_image_vgg_dir = base_dir + '/val/test_vgg/'
        self.test_result_file = base_dir + '/val/test_results' + self.st + '.txt'
        self.test_temp_file = base_dir + '/val/temp_test.csv'
        
        # about the evaluation
        self.eval_image_dir = base_dir + '/val/images_redo/'
        self.eval_image_vgg_dir = base_dir + '/val/images_vgg_redo/'
        self.eval_result_file = base_dir + '/val/val_results' + self.st + '.json'
        self.eval_temp_file = base_dir + '/val/temp_ann.csv'
        self.temp_generate_eval_file =  base_dir + '/val/temp_generate.npy'
        self.temp_image_file_eval = base_dir + '/val/image_files.npy'
        self.temp_feature_file_eval = base_dir + '/val/feature_files.txt'
        self.eval_temp_data_file = base_dir + '/val/data.npy'
        self.temp_eval_id = base_dir + '/val/ids.npy'
        self.num_eval_samples = 708

        self.trainable_variable = False
