import  pandas as pd #using pandas==0.25
import numpy as np
import sys
import os
import json
import pickle
from collections import defaultdict

import io
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter('ignore')
import csv

from helpers.Encoders import get_encoder
from helpers.Bucketers import get_bucketer
from helpers.DatasetManager import DatasetManager
import Definitions



train_ratio = 0.8
random_state = 22
min_cases_for_training = 1



encoding_dict = {
    'agg' : ['static', 'agg'],
    'index' : ['static', 'index'],
}


for method_name in ['single_agg', 'prefix_index']:
    
    bucket_method , cls_encoding = method_name.split('_')
    if bucket_method == "state":
        bucket_encoding = "last"
    else:
        bucket_encoding = "agg"
        
    if method_name == 'prefix_index':
      gap = 5
    encoded_datasets_dir = 'encoded_datasets_%s' %(method_name)
    if not os.path.exists(encoded_datasets_dir):
      os.makedirs(os.path.join(encoded_datasets_dir))
    
    datasets = ["sepsis2","traffic_fines","production", "BPIC2017_O_Accepted"]
    
    methods = encoding_dict[cls_encoding]
    
    outfile = os.path.join(encoded_datasets_dir,'all_datasets_info.csv')
    with open(outfile, 'w') as out:
            out.write('%s;%s;%s;%s;%s;%s\n' % ('dataset', 'method', 'dataset_type', 'bucket_size', 'prefix_length', 'feature_num'))
    
    for dataset_name in datasets:
    
        dm = DatasetManager(dataset_name)
        df = dm.read_dataset()
    
        cls_encoder_args_final = {'case_id_col': dm.case_id_col,
                                  'static_cat_cols': dm.static_cat_cols,
                                  'dynamic_cat_cols': dm.dynamic_cat_cols,
                                  'static_num_cols': dm.static_num_cols,
                                  'dynamic_num_cols': dm.dynamic_num_cols,
                                  'fillna': True}
    
        # determine min and max (truncated) prefix lengths
        min_prefix_length_final = 1
        if "traffic_fines" in dataset_name:
            max_prefix_length_final = 10
        elif "BPIC2017" in dataset_name:
            max_prefix_length_final = min(20, dm.get_pos_case_length_quantile(df, 0.90))
        else:
            max_prefix_length_final = min(40, dm.get_pos_case_length_quantile(df, 0.90))
    
        train, test = dm.split_data_strict(df, train_ratio=0.8, split='temporal')
    
        if (method_name == 'prefix_index'):
            df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final, gap=5)
            df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final, gap=5)
        else:
            df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final)
            df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final)
        
    
        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dm.case_id_col,
                         'cat_cols': [dm.activity_col],
                         'num_cols': [],
                         'random_state': random_state}
    
        bucketer = get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(df_train_prefixes)
        bucket_assignment_test = bucketer.predict(df_test_prefixes)
    
        nr_events_all = []
        
        for bucket in set(bucket_assignment_test):
            relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
            relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
            train_case_ids = list(relevant_train_bucket.values)
            test_case_ids = list(relevant_test_bucket.values)
            idx_test_set = [i for i in range(0,len(test_case_ids))]
            
            df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)
            test_prfx_len = dm.get_prefix_lengths(df_test_bucket)[0]
            test_y = np.array([dm.get_label_numeric(df_test_bucket)])
            nr_events_all.extend(list(dm.get_prefix_lengths(df_test_bucket)))
            if len(relevant_train_bucket) == 0:
                preds = [dm.get_class_ratio(train)] * len(relevant_test_bucket)
                current_online_times.extend([0] * len(preds))
            else:
                df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
                train_y_experiment = np.array([dm.get_label_numeric(df_train_bucket)])
                prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]
    
                
                featureCombinerExperiment = FeatureUnion(
                        [(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])
    
                encoded_training = featureCombinerExperiment.fit_transform(df_train_bucket)
                ffeatures = featureCombinerExperiment.get_feature_names()
                feat_num = len(ffeatures)
                ffeatures.append('encoded_label')
                
                
                
              
                encoded_training = np.concatenate((encoded_training,train_y_experiment.T), axis=1)
               
                training_set_df = pd.DataFrame(encoded_training, columns=ffeatures) 
                
                bkt_size = training_set_df.shape[0]
    
                encoded_testing_bucket = featureCombinerExperiment.fit_transform(df_test_bucket)
              
                encoded_testing_bucket = np.concatenate((encoded_testing_bucket,test_y.T), axis=1)
             
                testing_set_df = pd.DataFrame(encoded_testing_bucket, columns=ffeatures)
                
                test_bkt_size = testing_set_df.shape[0]
                
                
                # Save (serialize)
                pickle.dump(featureCombinerExperiment, open(os.path.join(encoded_datasets_dir,  "featureCombiner_%s_%s_%s_%s_%s.pkl" %(dataset_name, method_name, bkt_size, prfx_len, feat_num)), "wb" ) )
                
                
                matching_idx_rows = dict(zip(test_case_ids, idx_test_set))
                pickle.dump(matching_idx_rows, open(os.path.join(encoded_datasets_dir,  "matching_idx_rows_%s_%s_%s_%s_%s.pkl"% (dataset_name, method_name, bkt_size, prfx_len, feat_num)), "wb" ) )
                
                
                training_set_df.to_csv(os.path.join(encoded_datasets_dir, 'encoded_training_%s_%s_%s_%s_%s.csv' % (
                dataset_name, method_name, bkt_size, prfx_len, feat_num)), sep=';', columns= ffeatures, index=False)
                
                
                with open(outfile, 'a') as out:
                        out.write(
                            '%s;%s;%s;%s;%s;%s\n' % (dataset_name, method_name, 'training', bkt_size, prfx_len, feat_num))
                testing_set_df.to_csv(os.path.join(encoded_datasets_dir, 'encoded_testing_%s_%s_%s_%s_%s.csv' % (
                    dataset_name, method_name, test_bkt_size, test_prfx_len, feat_num)), sep=';', columns=ffeatures,
                                          index=False)
                with open(outfile, 'a') as out:
                        out.write('%s;%s;%s;%s;%s;%s\n' % (
                        dataset_name, method_name, 'testing', test_bkt_size, test_prfx_len, feat_num))

                
                whole_df  = pd.DataFrame(np.concatenate((encoded_training, encoded_testing_bucket), axis=0),
                            columns=ffeatures)
                            
                whole_df.to_csv(os.path.join(encoded_datasets_dir, 'encoded_wholeDF_%s_%s_%s_%s_%s.csv' % (
                dataset_name, method_name, bkt_size, prfx_len, feat_num)), sep=';', columns= ffeatures, index=False)

                
                print('done now with %s' %(dataset_name))
