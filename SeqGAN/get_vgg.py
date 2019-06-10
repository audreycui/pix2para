from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.models import Model
from utils.misc import ImageLoader
from keras.preprocessing import image
import pandas as pd
import numpy as np
import csv
from os import listdir

#computes feature map for images in the dataset
#feature map is computed as the output of the penultimate layer of vgg19

def get_feats(dir = False):

	train_caption_file = 'D:/download/art_desc/train/ann.csv'
	train_features_dir = 'D:/download/art_desc/train/images_vgg_redo/'

	eval_caption_file = 'D:/download/art_desc/val/ann.csv'
	#eval_image_dir = 'D:/download/art_desc/val/test_images/'
	eval_image_dir = 'D:/download/art_desc/val/images_redo/'
	
	eval_features_dir = 'D:/download/art_desc/val/images_vgg_redo/'
	#eval_features_dir = 'D:/download/art_desc/val/test_vgg/'
	image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')

	ignore_file = 'D:/download/art_desc/val/ignore.csv'

	net = VGG19(weights='imagenet')
	model = Model(input= net.input, output= net.get_layer('fc2').output)

	bad_ids = []
	prev_id = 0
	prev_bad = False

	if dir:
		with open(eval_caption_file, 'r') as f: #caption file
			reader = csv.reader(f)
			for id, file_name, caption in reader:
				try: 
				
					img = image_loader.load_image(file_name)
					'''
					fc2 = model.predict(img)
					reshaped = np.reshape(fc2, (4096))
					np.save(eval_features_dir + 'art_desc'+ str(id), reshaped)
					'''
					prev_bad = False
				except Exception:
					if id != prev_id or prev_bad is False:
						print ("cannot identify image file:" + file_name)
						bad_ids.append(id)
					prev_bad = True
				prev_id = id
			

	else:
		with open(train_caption_file, 'r') as f: #caption file
			reader = csv.reader(f)
			for id, file_name, caption in reader:
				try: 
					img = image_loader.load_image(file_name)
					'''
					fc2 = model.predict(img)
					reshaped = np.reshape(fc2, (4096))
					np.save(train_features_dir + 'art_desc'+ id, reshaped) #feature dir
					'''
					prev_bad = False
				except Exception:
					if id != prev_id or prev_bad is False:
						print ("cannot identify image file:" + file_name)
						bad_ids.append(id)
					prev_bad = True
					
				prev_id = id

	print("Total bad image files:%d" % len(bad_ids))
	data = pd.DataFrame({'index': bad_ids})
	data.to_csv(ignore_file)

get_feats(dir = True)