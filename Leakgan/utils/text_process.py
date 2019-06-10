# coding=utf-8
import os
import nltk
import numpy as np
import pandas as pd
import csv
import config
from utils.coco.coco import COCO
from utils.vocabulary import Vocabulary

#added functions for preparing test and validation data
def chinese_process(filein, fileout):
    with open(filein, 'r') as infile:
        with open(fileout, 'w') as outfile:
            for line in infile:
                output = list()
                line = nltk.word_tokenize(line)[0]
                for char in line:
                    output.append(char)
                    output.append(' ')
                output.append('\n')
                output = ''.join(output)
                outfile.write(output)


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file) as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict

def build_vocabulary(config, captions):
    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size, config.ctrl_symbols)
    if not os.path.exists(config.vocabulary_file):
        vocabulary.build(captions)
        vocabulary.save(config.vocabulary_file)
    else:
        vocabulary.load(config.vocabulary_file)
    #print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))
    #return vocabulary

    print("NUM CAPTIONS: " + str(len(captions)))
    data_file = config.temp_data_file
    
    if not os.path.exists(data_file):
        word_idxs = []
        sent_lens = []
        for caption in captions:
            current_word_idxs, current_length = vocabulary.process_sentence(caption)
            current_num_words = min(config.max_caption_length-2, current_length)

            pad_length = config.max_caption_length - current_length - 2
            current_word_idxs = [config._START_] + current_word_idxs[:current_num_words] + [config._END_] + [config._PAD_] * pad_length

            word_idxs.append(current_word_idxs)
            sent_lens.append(current_num_words+2)
        word_idxs = np.array(word_idxs)
        print("generating temp file sentence length " + str(len(word_idxs)))
        data = {'word_idxs': word_idxs, 'sentence_len': sent_lens}
        np.save(data_file, data)
    else:
        data = np.load(data_file).item()
        word_idxs = data['word_idxs']
        sent_lens = data['sentence_len']
    '''
    if oracle_file is not None:
        with open(oracle_file, 'w') as outfile:
            paras = ""
            for line in word_idxs:
                for word in line:
                    paras += (str(word) + ' ')
                paras += '\n'
                outfile.write(paras)
    '''

    return vocabulary

def text_precess(train_text_loc, test_text_loc=None, oracle_file=None):
    train_tokens = get_tokenlized(train_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))

    with open('save/eval_data.txt', 'w') as outfile:
        outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))

    if oracle_file is not None:
        with open(oracle_file, 'w') as outfile:
            outfile.write(text_to_code(train_tokens, word_index_dict, sequence_len))

    return sequence_len, len(word_index_dict) + 1

def process_text_only(config, data_loc, oracle_file):
    captions = [line.rstrip('\n') for line in open(data_loc)] 
    vocabulary = build_vocabulary(config, captions)

    return config.max_caption_length, vocabulary.size + len(config.ctrl_symbols), vocabulary

def process_train_data(config, data_loc, orcale_file=None, has_image=False, all_captions = True):
    if data_loc is None:
        data_loc = 'data/caption.txt'
    if not has_image:
        return process_text_only(config, data_loc, orcale_file)

    print("Processing the captions...")
    if not os.path.exists(config.temp_annotation_file):
        print("Creating temp annotation file....")
        if all_captions:
            print("ALL CAPTIONS")
            captions = []
            image_ids = []
            image_files = []
            feature_files = []
            ignore_ids = []
            #if ignore_file:

            df = pd.read_csv(config.ignore_file).values
            ignore_ids = [idx for seqno, idx in df]

            ann_file = pd.read_csv(config.train_caption_file)
            with open(config.train_caption_file, 'r') as f:
                reader = csv.reader(f)
                for id, file_name, caption in reader:
                    if int(id) not in ignore_ids:
                        image_ids.append(id)
                        image_files.append(file_name)
                        feature_files.append(os.path.join(config.train_feature_dir,
                                        os.path.basename(file_name.replace('.jpg', '.npy'))))
                        captions.append(caption)
            annotations = pd.DataFrame({'image_id': image_ids,
                                        'image_file': image_files,
                                        'feature_file': feature_files,
                                        'caption': captions})
            annotations.to_csv(config.temp_annotation_file)

            all_captions = captions
        else:   
            print("NOT ALL CAPTIONS")   
            coco = COCO(config.train_caption_file, config.ignore_file)
            all_captions = coco.all_captions
            captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
            image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
            image_files = [os.path.join(config.train_image_dir,
                                        coco.imgs[image_id]['file_name'])
                                        for image_id in image_ids]
            feature_files = [os.path.join(config.train_feature_dir,
                                        os.path.basename(coco.imgs[image_id]['file_name'].replace('.jpg', '.npy')))
                                        for image_id in image_ids]
            annotations = pd.DataFrame({'image_id': image_ids,
                                        'image_file': image_files,
                                        'feature_file': feature_files,
                                        'caption': captions})
            annotations.to_csv(config.temp_annotation_file)

        print(len(image_ids), len(image_files), len(feature_files), len(captions))
    else:
        annotations = pd.read_csv(config.temp_annotation_file)
        captions = [] 
        image_ids = [] 
        image_files = [] 
        feature_files = []
        for _, id, file, feature, cap in annotations.values:
            image_ids.append(id)
            image_files.append(file)
            feature_files.append(feature)
            captions.append(cap)
        print("load data...")
        print(len(image_ids), len(image_files), len(feature_files), len(captions))

        all_captions = captions
    with open(config.temp_image_file, 'w') as outfile:
        for img_file in image_files:
            outfile.write(img_file+"\n")
    with open(config.temp_feature_file, 'w') as outfile:
        for feature in feature_files:
            outfile.write(feature+"\n")

    vocabulary = build_vocabulary(config, all_captions)

    return config.max_caption_length, vocabulary.size + len(config.ctrl_symbols), vocabulary

def process_test_data(config):
    #vocabulary = Vocabulary(config.vocabulary_size, config.ctrl_symbols)
    coco = COCO(config.train_caption_file, config.ignore_file)

    vocabulary = build_vocabulary(config, coco.all_captions())
    if not os.path.exists(config.test_temp_file):
        image_files = [config.test_image_dir+f for f in os.listdir(config.test_image_dir)]
        feature_files = [config.test_image_vgg_dir+f for f in os.listdir(config.test_image_vgg_dir)]
        data = pd.DataFrame({'image_file': image_files, 'feature_file': feature_files})
        data.to_csv(config.test_temp_file)

    return config.max_caption_length, vocabulary.size + len(config.ctrl_symbols), vocabulary

def process_val_data(config):
    #vocabulary = Vocabulary(config.vocabulary_size, config.ctrl_symbols)
    coco = COCO(config.train_caption_file, config.ignore_file)

    vocabulary = build_vocabulary(config, coco.all_captions())


    coco = COCO(config.eval_caption_file, config.ignore_file_eval)

    all_captions = coco.all_captions
    if not os.path.exists(config.eval_temp_file):
        df = pd.read_csv(config.ignore_file_eval).values
        ignore_ids = [int(idx) for seqno, idx in df]
        captions = []
        image_ids = []
        for ann_id in coco.anns: 
            if int(ann_id) not in ignore_ids:
                #print(ann_id)
                captions.append(coco.anns[ann_id]['caption'])
                image_ids.append(coco.anns[ann_id]['image_id'])
        image_files = [os.path.join(config.train_image_dir,
                                    coco.imgs[image_id]['file_name'])
                                    for image_id in image_ids]
        feature_files = [os.path.join(config.train_feature_dir,
                                    os.path.basename(coco.imgs[image_id]['file_name'].replace('.jpg', '.npy')))
                                    for image_id in image_ids]
        annotations = pd.DataFrame({'image_id': image_ids,
                                    'image_file': image_files,
                                    'feature_file': feature_files,
                                    'caption': captions})
        annotations.to_csv(config.eval_temp_file)

        data = pd.DataFrame({'image_file': image_files, 'feature_file': feature_files})
        data.to_csv(config.eval_temp_file)

    return config.max_caption_length, vocabulary.size + len(config.ctrl_symbols), vocabulary

def process_val_data_old(config, orcale_file=None, has_image=False, all_captions = True):

    print("Processing the captions...")
    if not os.path.exists(config.eval_temp_file):
        print("Creating temp annotation file....")
        if all_captions:
            print("ALL CAPTIONS")
            captions = []
            image_ids = []
            image_files = []
            feature_files = []
            ignore_ids = []
            #if ignore_file:

            df = pd.read_csv(config.ignore_file_eval).values
            ignore_ids = [idx for seqno, idx in df]

            ann_file = pd.read_csv(config.eval_caption_file)
            with open(config.eval_caption_file, 'r') as f:
                reader = csv.reader(f)
                for id, file_name, caption in reader:
                    if int(id) not in ignore_ids:
                        image_ids.append(id)
                        image_files.append(file_name)
                        feature_files.append(os.path.join(config.train_feature_dir,
                                        os.path.basename(file_name.replace('.jpg', '.npy'))))
                        captions.append(caption)
            annotations = pd.DataFrame({'image_id': image_ids,
                                        'image_file': image_files,
                                        'feature_file': feature_files,
                                        'caption': captions})
            annotations.to_csv(config.eval_temp_file)

            all_captions = captions
        else:   
            print("NOT ALL CAPTIONS")   
            coco = COCO(config.eval_caption_file, config.ignore_file_eval)
            all_captions = coco.all_captions
            captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
            image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
            image_files = [os.path.join(config.train_image_dir,
                                        coco.imgs[image_id]['file_name'])
                                        for image_id in image_ids]
            feature_files = [os.path.join(config.train_feature_dir,
                                        os.path.basename(coco.imgs[image_id]['file_name'].replace('.jpg', '.npy')))
                                        for image_id in image_ids]
            annotations = pd.DataFrame({'image_id': image_ids,
                                        'image_file': image_files,
                                        'feature_file': feature_files,
                                        'caption': captions})
            annotations.to_csv(config.eval_temp_file)

        print(len(image_ids), len(image_files), len(feature_files), len(captions))
    else:
        annotations = pd.read_csv(config.eval_temp_file)
        captions = [] 
        image_ids = [] 
        image_files = [] 
        feature_files = []
        for _, id, file, feature, cap in annotations.values:
            image_ids.append(id)
            image_files.append(file)
            feature_files.append(feature)
            captions.append(cap)
        print("load data...")
        print(len(image_ids), len(image_files), len(feature_files), len(captions))

        all_captions = captions
    with open(config.temp_image_file_eval, 'w') as outfile:
        for img_file in image_files:
            outfile.write(img_file+"\n")
    with open(config.temp_feature_file_eval, 'w') as outfile:
        for feature in feature_files:
            outfile.write(feature+"\n")

    coco = COCO(config.train_caption_file, config.ignore_file)

    vocabulary = build_vocabulary(config, coco.all_captions())

    return config.max_caption_length, vocabulary.size + len(config.ctrl_symbols), vocabulary
