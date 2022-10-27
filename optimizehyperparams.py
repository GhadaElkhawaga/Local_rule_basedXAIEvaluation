import pandas as pd
import numpy as np
import os
import csv
import pickle
import time
from collections import defaultdict
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from rulefit import RuleFit
import xgboost as xgb
from helpers.DatasetManager import DatasetManager
from helpers.Bucketers import get_bucketer
from helpers.Encoders import get_encoder
from utils.retrieval import retrieve_artefact


#Defining Basic parameters
train_ratio = 0.8
n_splits = 3
random_state = 22
n_iter = 5
encoding_dict = {"agg": ["static", "agg"], "index": ["static", "index"]}
artefacts_dir = 'localXAIEval_October22'
if not os.path.exists(artefacts_dir):
  os.makedirs(os.path.join(artefacts_dir))

params_dir = os.path.join(artefacts_dir,'cv_results_Oct22')
if not os.path.exists(params_dir):
  os.makedirs(os.path.join(params_dir))
  

dataset_ref = "sepsis_cases"
dataset_ref_to_datasets = {"sepsis_cases": [ "sepsis3"]}
datasets = ['sepsis2', 'sepsis3', 'traffic_fines', 'production',  "BPIC2017_O_Accepted"]

  
# hyperparameter Optimization objective function
def create_and_evaluate_model(args):
    global trial_nr
    trial_nr += 1

    start = time.time()
    score = 0
    for cv_iter in range(n_splits):

        df_test_prefixes = df_prefixes[cv_iter]
       
        df_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits):
            if cv_train_iter != cv_iter:
                df_train_prefixes = pd.concat([df_train_prefixes, df_prefixes[cv_train_iter]], axis=0)
            
        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dataset_manager.case_id_col,
                         'cat_cols': [dataset_manager.activity_col],
                         'num_cols': [],
                         'random_state': random_state}

        bucketer = get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(df_train_prefixes)
        bucket_assignments_test = bucketer.predict(df_test_prefixes)

        preds_all = []
        test_y_all = []
        if "prefix" in method_name:
            scores = defaultdict(int)
        for bucket in set(bucket_assignments_test):
            relevant_train_cases_bucket = dataset_manager.get_indexes(df_train_prefixes)[
                bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(df_test_prefixes)[
                bucket_assignments_test == bucket]
            df_test_bucket = dataset_manager.get_relevant_data_by_indexes(df_test_prefixes, relevant_test_cases_bucket)
            test_y = dataset_manager.get_label_numeric(df_test_bucket)
            
            if len(relevant_train_cases_bucket) == 0:
                preds = [class_ratios[cv_iter]] * len(relevant_test_cases_bucket)
            else:
                df_train_bucket = dataset_manager.get_relevant_data_by_indexes(df_train_prefixes,
                                                                               relevant_train_cases_bucket)  # one row per event
                
                train_y = dataset_manager.get_label_numeric(df_train_bucket)

                if len(set(train_y)) < 2:
                    preds = [train_y[0]] * len(relevant_test_cases_bucket)
                else:
                    feature_combiner = FeatureUnion(
                        [(method, get_encoder(method, **cls_encoder_args)) for method in methods])
                    #df_train_prefixes = feature_combiner.fit_transform(df_train_prefixes, train_y)
                    #df_test_prefixes = feature_combiner.fit_transform(df_test_prefixes, test_y)
                    if cls_method == "xgboost":
                        cls = xgb.XGBClassifier(objective='binary:logistic',
                                                n_estimators=500,
                                                learning_rate=args['learning_rate'],
                                                subsample=args['subsample'],
                                                max_depth=int(args['max_depth']),
                                                colsample_bytree=args['colsample_bytree'],
                                                min_child_weight=int(args['min_child_weight']),
                                                seed=random_state)

                    elif cls_method == "logit":
                        cls = LogisticRegression(C=2 ** args['C'],
                                                 random_state=random_state)
                    
                    elif cls_method == 'rulefit':
                        """cls = RuleFit(rfmode='classify', tree_generator=
                                                        GradientBoostingClassifier(n_estimators=500, random_state=22, 
                                                        learning_rate=args['learning_rate']))  """
                        cls = RuleFit(tree_size=args['tree_size'], sample_fract='default', max_rules=2000, memory_par=0.01,
                                        tree_generator=None, rfmode='classify', lin_trim_quantile=args['lin_trim_quantile'], 
                                        lin_standardise=True, exp_rand_tree_size=True, random_state=1)
                    
                    elif cls_method == 'rf':
                        cls = RandomForestClassifier(n_estimators=500,
                                                     max_features=args['max_features'],
                                                     random_state=random_state)
                    
                    elif cls_method == 'mlp':
                       input_shape = df_train_bucket.shape
                       size1 = (input_shape[1]+2)//2
                       size2 = int(input_shape[0]/(2*(input_shape[1]+2)))
                       
                       cls = MLPClassifier(hidden_layer_sizes =  (size2,),
                                           activation = args['activation'], solver = args['solver'],
                                           learning_rate=args['learning_rate'], max_iter = args['max_iter'], alpha=args['alpha'])
                       #hp.choice('hidden_layer_sizes', [(size1,), (size2,)]), 
                       
                   
                    if cls_method in ["logit", 'mlp']:
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    else:
                        pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
                        
                    pipeline.fit(df_train_bucket, train_y)
                    if cls_method == 'rulefit':
                        preds = pipeline.predict(df_test_bucket)
                    else:
                        preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                        preds = pipeline.predict_proba(df_test_bucket)[:, preds_pos_label_idx]

            if "prefix" in method_name:
                auc = 0.5
                if len(set(test_y)) == 2:
                    auc = roc_auc_score(test_y, preds)
                scores[bucket] += auc
            preds_all.extend(preds)
            test_y_all.extend(test_y)

        # score += roc_auc_score(test_y_all, preds_all)
        try:
            score += roc_auc_score(test_y_all, preds_all)
        except ValueError:
            pass

    if "prefix" in method_name:
        for k, v in args.items():
            for bucket, bucket_score in scores.items():
                fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                trial_nr, dataset_name, cls_method, method_name, bucket, k, v, bucket_score / n_splits))
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
        trial_nr, dataset_name, cls_method, method_name, 0, "processing_time", time.time() - start, 0))
    else:
        for k, v in args.items():
            fout_all.write(
                "%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (
        trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))
    fout_all.flush()
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


# code to find the best parameters
for dataset_name in datasets:
    for cls_method in ['logit', 'mlp', 'xgboost', 'rf', 'rulefit']:
      for method_name in ['prefix_index', 'single_agg']:
            print('%s_%s_%s' %(dataset_name, cls_method, method_name))
            if method_name == "single_agg":
              gap = 1
            else:
              gap = 5
            bucket_method , cls_encoding = method_name.split('_')
            if bucket_method == "state":
                bucket_encoding = "last"
            else:
                bucket_encoding = "agg"
            methods = encoding_dict[cls_encoding]

            # the folders that contains the folds csv files
            folds_directory = os.path.join(artefacts_dir, 'folds_%s_%s_%s' % (dataset_name, cls_method, method_name))
            if not os.path.exists(os.path.join(folds_directory)):
                os.makedirs(os.path.join(folds_directory))
        
            dataset_manager = DatasetManager(dataset_name)
            df = dataset_manager.read_dataset()
        
            cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                                'static_cat_cols': dataset_manager.static_cat_cols,
                                'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                                'static_num_cols': dataset_manager.static_num_cols,
                                'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                                'fillna': True}
        
            # determine min and max (truncated) prefix lengths
            min_prefix_length = 1
            if "traffic_fines" in dataset_name:
                max_prefix_length = 10
            elif "BPIC2017" in dataset_name:
                max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(df, 0.90))
            else:
                max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(df, 0.90))
        
            # splitting data into training and testing, then deleting the whole dataframe
            train, _ = dataset_manager.split_data_strict(df, train_ratio, split="temporal")
            del df
           
            # prepare chunks for cross-validation
            df_prefixes = []
            class_ratios = []
            for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
                class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
                # generate data where each prefix is a separate instance
                if (method_name == 'prefix_index'):
                    df_prefixes.append(
                        dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length, gap=5))
                else:
                    df_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
            del train
        
            # search parameters for 'xgboost' algorithm:
            if cls_method == 'xgboost':
                space = {'learning_rate': hp.uniform('learning_rate', 0, 1),
                         'subsample': hp.uniform('subsample', 0.5, 1),
                         'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                         'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
            elif cls_method == 'rf':
                space = {'max_features': hp.uniform('max_features', 0, 1)}  
            elif cls_method == 'rulefit':
                #space = {'learning_rate': hp.uniform("learning_rate", 0, 1)}  
                space = {"tree_size":hp.choice('tree_size', [4,8,12,16]),
                 "lin_trim_quantile": hp.uniform('lin_trim_quantile', 0.025, 1)}
            elif cls_method == 'logit':
                space = {'C': hp.uniform('C', -15, 15)}
            elif cls_method == 'mlp':
                space = { 'activation': hp.choice('activation',['tanh', 'relu']), 'solver': hp.choice('solver',['sgd', 'adam']), 
                         'learning_rate': hp.choice('learning_rate', ['constant','adaptive']),
                         'max_iter': hp.choice('max_iter', [50, 100, 150, 200,250, 300, 500, 600]), 'alpha': hp.choice('alpha', [0.0001, 0.05]) }
        
            trial_nr = 1
            # trial_nr = 0
            trials = Trials()
        
            fout_all = open(
                os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
            if "prefix" in method_name:
                fout_all.write(
                    "%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "score"))
            else:
                fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "score"))
            rstate = np.random.RandomState(22)
            best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials, rstate = rstate)
           
            fout_all.close()
        
            # write the best parameters
            best_params = hyperopt.space_eval(space, best)
            outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
            # write to file
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)
                
                