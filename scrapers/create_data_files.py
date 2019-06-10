import numpy
from os import listdir
from numpy import genfromtxt
#from textteaser.textteaser.teaser import TextTeaser
from pyteaser import Summarize
import cv2
import csv
import nltk

smith_dir = 'D:/download/smithsonian'
story_dir = 'D:/download/artstory/'
train_img_dir = 'D:/download/art_desc/train/images_redo'
val_img_dir = 'D:/download/art_desc/val/images_redo'
train_ann_dir = 'D:/download/art_desc/train/ann.csv'
val_ann_dir = 'D:/download/art_desc/val/ann.csv'

global counter 

#splits artwork-description pairs into 80% train, 20% val
def create_file(base_dir):
	pair_dir = listdir(base_dir)
	#train 80%, cal 20%
	counter = 0
	for directory in pair_dir: 
		
		print(counter)
		dir_path = base_dir+'/'+directory
		files = listdir(dir_path)
		if len(files) < 2:
		    continue
		image_file = dir_path + '/' + files[0]
		text_file = dir_path + '/' + files[1]
		try:
		#if True:
			read_text = open (text_file, 'r')
			full_text = read_text.read()
			full_text = nltk.sent_tokenize(full_text)
		   

			text_dir = ''
			img = cv2.imread(image_file)
			img_dir = ''

			if counter%5 == 0: #20% val, 80% train
				img_dir = val_img_dir+'/'+'art_desc'+str(counter)+'.jpg'
				cv2.imwrite(img_dir, img)
				text_dir = val_ann_dir

			else: 
				img_dir = train_img_dir+'/'+'art_desc'+str(counter)+'.jpg'
				cv2.imwrite(img_dir, img)
				text_dir = train_ann_dir

			for text in full_text:
				ann = [counter, img_dir, text]
				with open(text_dir, 'a', newline = '') as f:
					writer = csv.writer(f)
					writer.writerow(ann)
			counter+=1

		except:
			print ("error while processing, continue")
			print (text_file, image_file)
create_file(story_dir)
create_file(smith_dir)