from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def language_eval_chinese(dataset, preds, model_id, split):
    from caption_eval.coco_caption.pycxtools.coco import COCO
    from caption_eval.coco_caption.pycxevalcap.eval import COCOEvalCap
    m1_score = {}
    m1_score['error'] = 0

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    reference_file = "/home/hc/image2txt/chinese_im2text.pytorch/vis/reference.json" #'/home/hc/image_root/ai_challenger_caption_validation_20170910/coco_val_caption_validation_annotations_20170910.json'
    # "/home/hc/image2txt/chinese_im2text.pytorch/vis/reference.json"
    coco = COCO(reference_file)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    coco_res = coco.loadRes(cache_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    # evaluate results
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print('%s: %.3f' % (metric, score))
        m1_score[metric] = score

    return m1_score

reference = "/home/hc/image2txt/chinese_im2text.pytorch/vis/reference.json"
preds = "/home/hc/image2txt/chinese_im2text.pytorch/vis/val_submit1101_1058.json"
test_pred = json.load(open(preds))

aaa = language_eval_chinese(11,test_pred,"test","vald")