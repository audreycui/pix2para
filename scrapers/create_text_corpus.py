#creates a text corpus of all scraped text
#removes sentences with length >50

import numpy
from os import listdir
from numpy import genfromtxt
#from textteaser.textteaser.teaser import TextTeaser

import cv2
import csv
from nltk.tokenize import sent_tokenize, word_tokenize

smith_dir = 'D:/download/smithsonian/'
story_dir = 'D:/download/artstory/'
corpus_dir = 'D:/download/art_desc/full_corpus.txt'

global counter 


def create_file(base_dir):
	pair_dir = listdir(base_dir)
	
	appender = open(corpus_dir, 'a')
	for directory in pair_dir: 
		
		print(directory)
		dir_path = base_dir+'/'+directory
		files = listdir(dir_path)
		if len(files) < 2:
		    continue
		
		text_file = dir_path + '/' + files[1]

		try:
			read_text = open(text_file, 'r')
			text = read_text.read()
			sentences = sent_tokenize(text)
			for sent in sentences: 
				tokens = word_tokenize(sent)
				if len(tokens) < 51: 
					appender.write(sent.strip()+'\n')
		except:
			print ("error while processing, continue")
			
	appender.close()

create_file(story_dir)
create_file(smith_dir)