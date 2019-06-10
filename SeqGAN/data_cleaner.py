#!/usr/bin/python
import tensorflow as tf

from scipy.misc import imread, imresize
from imagenet_classes import class_names
import numpy as np
import pandas as pd
import pprint
import json 
import csv
import sys
import os

from config import Config
from keras.preprocessing import image
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary
from utils.misc import ImageLoader

#checks for corrputed image files
#records ids of corrputed image files in an a csv file


def main():
    config = Config()

    #load_ignore_file(config)
    #prepare_train_data(config)
    cleanup_data(config)

def load_ignore_file(config):
  df = pd.read_csv(config.ignore_file).values
  df = [idx for seqno, idx in df]
  print(df)

def cleanup_data(config):
    bad_ids = []
    try:
        with open(config.train_caption_file, 'r') as f:
            self.dataset = json.load(f)
    except Exception:
        #try:
        with open(config.train_caption_file, 'r') as f:
            reader = csv.reader(f)
            for id, file_name, caption in reader:
                try:
                    img = image.load_img(file_name, target_size=(224, 224))
                except Exception:
                    print ("cannot identify image file:" + file_name)
                    bad_ids.append(id)
                    pass
    
        #except Exception:
            #print ("Unsupported caption file format other than json or csv")
            #return
        
    print("Total bad image files:%d" % len(bad_ids))
    data = pd.DataFrame({'index': bad_ids})
    data.to_csv(config.ignore_file)

if __name__ == '__main__':
    main()