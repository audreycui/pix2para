import pandas as pd 
import csv
import numpy as np
import sys
import json
import datetime
import time
#formats validation data (image id, generated caption, reference captions) 
#into a json file that can be inputted for SPICE evaluation

def prepare_json(config):
    generated_samples = np.load(config.temp_generate_eval_file)
    ids = np.load(config.temp_eval_id)
    ignore_ids = []
    df = pd.read_csv(config.ignore_file_eval).values
    ignore_ids = [idx for seqno, idx in df]

    ann_file = pd.read_csv(config.eval_caption_file)
    all_refs = []
    id_idx = 0
    prev_id = ids[id_idx]
    ref = []
    with open(config.eval_caption_file, 'r') as f:
        reader = csv.reader(f)
        for id, file, caption in reader:
            try: 
                prev_id = ids[id_idx]
            except IndexError: 
            	break
            if int(id) not in ignore_ids:
                #print(str(id))
                if (int(id)==prev_id):
                    #print(id + " " + str(prev_id)+ " equal")
                    ref.append(caption)
                else:
                    #print(id + " " + str(prev_id)+ " unequal")
                    all_refs.append(ref)
                    ref = []
                    ref.append(caption)
                    id_idx+=1
                
    #print(all_refs)
    results = []
    for (id, real, gen) in zip(ids, all_refs, generated_samples):
        results.append({'image_id':str(id), 'test': gen, 'refs': real})

    fp = open(config.eval_result_file, 'w')
    json.dump(results, fp)
    fp.close()

