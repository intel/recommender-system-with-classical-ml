#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
#import xgboost as xgb
from xgboost_ray import RayDMatrix, RayParams, train, predict
from sklearn.metrics import log_loss, average_precision_score
from features import *
import yaml
from pathlib import Path
import shutil
import os
import sys
from ray import tune
import ray 

ray.init()
num_actors = 4
num_cpus_per_actor = 4
ray_params = RayParams(
    num_actors=num_actors, cpus_per_actor=num_cpus_per_actor)

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

        df_train = pd.read_parquet(glob.glob(f'{data_path}/train/stage1/train/*.parquet')[0])[:10000]
        df_valid = pd.read_parquet(glob.glob(f'{data_path}/train/stage1/valid/*.parquet')[0])[:10000]
    else:
        df_train = pd.read_parquet(f'{data_path}/train/stage1/train/')
        df_valid = pd.read_parquet(f'{data_path}/train/stage1/valid/')
    print(df_train.shape)
    print(df_valid.shape)

    for col in df_valid.columns:
        if df_valid[col].dtype=='bool':
            df_train[col] = df_train[col].astype('int8')
            df_valid[col] = df_valid[col].astype('int8')

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
    # xgb_parms = { 
    #     'max_depth':8, 
    #     'learning_rate':0.1, 
    #     'subsample':0.8,
    #     'colsample_bytree':0.8, 
    #     'eval_metric':'logloss',
    #     'objective':'binary:logistic',
    #     'tree_method':'hist',
    #     "random_state":42
    # }

    oof = np.zeros((len(df_valid),len(label_names)))
    for numlabel in range(4):
        start = time.time()
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)
        
        #dtrain = xgb.DMatrix(data=train[feature_list[numlabel]], label=train[name])
        #dvalid = xgb.DMatrix(data=valid[feature_list[numlabel]], label=valid[name])

        print("Training.....")
        # model = train(xgb_parms, 
        #         dtrain,
        #         evals=[(dtrain,'train'),(dvalid,'valid')],
        #         num_boost_round=250,
        #         early_stopping_rounds=25,
        #         #maximize=True,
        #         verbose_eval=25,
        #         ray_params=ray_params)

        def train_model(config):
            train_set = RayDMatrix(df_train[feature_list[numlabel]], df_train[name])
            valid_set = RayDMatrix(df_valid[feature_list[numlabel]], df_valid[name])

            evals_result = {}
            bst = train(
                params=config,
                dtrain=train_set,
                evals_result=evals_result,
                evals=[(train_set, "train"),(valid_set, "valid")],
                verbose_eval=False,
                ray_params=ray_params)
            bst.save_model(f"{model_save_path}/xgboost_{name}_stage1.model")

        
        config = {
                    # "tree_method": "hist",
                    # "objective": "binary:logistic",
                    # "learning_rate": tune.choice([0.1,0.4]),
                    # "eval_metric": ["logloss", "error"],
                    # "eta": tune.loguniform(1e-4, 1e-1),
                    # "subsample": tune.uniform(0.5, 1.0),
                    # "max_depth": tune.randint(1, 9)
                    "tree_method": "approx",
                    "objective": "binary:logistic",
                    "eval_metric": ["logloss", "error"],
                    "eta": tune.loguniform(1e-4, 1e-1),
                    "subsample": tune.uniform(0.5, 1.0),
                    "max_depth": tune.randint(1, 9)
                }

        #model.save_model(f"{model_save_path}/xgboost_{name}_stage1.model")

        analysis = tune.run(
                            train_model,
                            config=config,
                            metric="train-error",
                            mode="min",
                            num_samples=4,
                            resources_per_trial=ray_params.get_tune_resources()
                            )

        print("Best hyperparameters", analysis.best_config)


        # print('Predicting...')
        # oof[:,numlabel] = predict(model, dvalid,  ray_params=ray_params)
        
        # print("took %.1f seconds" % ((time.time()-start)))

    ######## Merge prediction to data and save
    # for i in range(4):
    #     df_valid[f"pred_{label_names[i]}"] = oof[:,i]
    
    # df_valid[["tweet_id","engaging_user_id",f"pred_{label_names[0]}",f"pred_{label_names[1]}",f"pred_{label_names[2]}",f"pred_{label_names[3]}"]].to_csv(f"{pred_save_path}/xgboost_pred_stage1.csv",index=0)
    
    ######## Evaluate the performance
    # txts = ''
    # sumap = 0
    # sumrce = 0
    # for i in range(4):
    #     ap = compute_AP(oof[:,i],df_valid[label_names[i]].values)
    #     rce = compute_rce_fast(oof[:,i],df_valid[label_names[i]].values)
    #     txt = f"{label_names[i]:20} AP:{ap:.5f} RCE:{rce:.5f}"
    #     print(txt)

    #     txts += "%.4f" % ap + ' '
    #     txts += "%.4f" % rce + ' '
    #     sumap += ap
    #     sumrce += rce
    # print(txts)
    # print("AVG AP: ", sumap/4.)
    # print("AVG RCE: ", sumrce/4.)
    
    # print('This notebook took %.1f seconds'%(time.time()-very_start))
