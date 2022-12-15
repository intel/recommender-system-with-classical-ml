#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, average_precision_score
from features import *
import yaml
from pathlib import Path
import shutil
import os
import sys

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
very_start = time.time()

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def check_test_path(model_save_path):
    if model_save_path.is_dir():
        print('WARNING: test model directory \'{}\' already exists - will be removed and newly created '.format(
                    model_save_path), sys.stderr)
        shutil.rmtree(model_save_path, ignore_errors=True)
        model_save_path.mkdir(parents=True)
    else:
        print(
            'WARNING: test model directory \'{}\' did not exist - will be newly created '.format(
                model_save_path), sys.stderr)
        model_save_path.mkdir(parents=True)
    return None

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..'))

try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")

data_path = config['files']['data_path'] 
model_save_path = config['files']['model_save_path'] 
pred_save_path = config['files']['pred_save_path'] 
num_iterations = config['training']['num_iterations']
DEBUG = config['training']['debug']

if __name__ == "__main__":
    ######## Load data
    if DEBUG:
        num_iterations = config['training']['num_iterations_debug'] 
        model_save_path = Path(os.path.join(model_save_path, 'test'))
        pred_save_path = Path(os.path.join(pred_save_path, 'test'))

        check_test_path(model_save_path)
        check_test_path(pred_save_path)

        train = pd.read_parquet(glob.glob(f'{data_path}/stage1_train/*.parquet')[0])[:10000]
        valid = pd.read_parquet(glob.glob(f'{data_path}/stage1_valid/*.parquet')[0])[:10000]
    else:
        train = pd.read_parquet(f'{data_path}/stage1_train/')
        valid = pd.read_parquet(f'{data_path}/stage1_valid/')
    print(train.shape)
    print(valid.shape)

    for col in valid.columns:
        if valid[col].dtype=='bool':
            train[col] = train[col].astype('int8')
            valid[col] = valid[col].astype('int8')

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    feature_list = []
    feature_list.append(stage1_reply_features)
    feature_list.append(stage1_retweet_features)
    feature_list.append(stage1_comment_features)
    feature_list.append(stage1_like_features)
    for i in range(4):
        print(len(feature_list[i]))

    ######## Train and predict
    xgb_parms = { 
        'max_depth':8, 
        'learning_rate':0.1, 
        'subsample':0.8,
        'colsample_bytree':0.8, 
        'eval_metric':'logloss',
        'objective':'binary:logistic',
        'tree_method':'hist',
        "random_state":42
    }

    oof = np.zeros((len(valid),len(label_names)))
    for numlabel in range(4):
        start = time.time()
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)
        
        dtrain = xgb.DMatrix(data=train[feature_list[numlabel]], label=train[name])
        dvalid = xgb.DMatrix(data=valid[feature_list[numlabel]], label=valid[name])

        print("Training.....")
        model = xgb.train(xgb_parms, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=250,
                early_stopping_rounds=25,
                #maximize=True,
                verbose_eval=25) 
        model.save_model(f"{model_save_path}/xgboost_{name}_stage1.model")

        print('Predicting...')
        oof[:,numlabel] = model.predict(dvalid)
        
        print("took %.1f seconds" % ((time.time()-start)))

    ######## Merge prediction to data and save
    for i in range(4):
        valid[f"pred_{label_names[i]}"] = oof[:,i]
    
    valid[["tweet_id","engaging_user_id",f"pred_{label_names[0]}",f"pred_{label_names[1]}",f"pred_{label_names[2]}",f"pred_{label_names[3]}"]].to_csv(f"{pred_save_path}/xgboost_pred_stage1.csv",index=0)
    
    ######## Evaluate the performance
    txts = ''
    sumap = 0
    sumrce = 0
    for i in range(4):
        ap = compute_AP(oof[:,i],valid[label_names[i]].values)
        rce = compute_rce_fast(oof[:,i],valid[label_names[i]].values)
        txt = f"{label_names[i]:20} AP:{ap:.5f} RCE:{rce:.5f}"
        print(txt)

        txts += "%.4f" % ap + ' '
        txts += "%.4f" % rce + ' '
        sumap += ap
        sumrce += rce
    print(txts)
    print("AVG AP: ", sumap/4.)
    print("AVG RCE: ", sumrce/4.)
    
    print('This notebook took %.1f seconds'%(time.time()-very_start))

