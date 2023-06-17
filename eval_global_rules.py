import re
import os
import sys
import time
import pickle
import logging
import fnmatch
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from mlxtend.plotting import plot_confusion_matrix

from rules_utils import save_pkl, get_logger
from rulefit import RuleFit
import wittgenstein as lw
##a warokaround to be able to import skrules 
import six
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules


from helpers.DatasetManager import DatasetManager
from helpers.Bucketers import get_bucketer
from helpers.Encoders import get_encoder
from utils.retrieval import retrieve_artefact

saved_artefacts = 'models'
logs = 'logs'
rules_dir = 'rules'
bb_preds = os.path.join('artefacts','predictions_truly_predicted')
discretized_logs = os.path.join('artefacts','discretized_encoded_training')
outputs = os.path.join('artefacts','predictions_all_vs_all_classifiers')


encoding_dict = {
    'agg' : ['static', 'agg'],
    'index' : ['static', 'index']}

    

def evaluate(bb_preds, ruleMpreds):
    
    acc = accuracy_score(bb_preds, ruleMpreds)
    f1 = f1_score(bb_preds, ruleMpreds)
    try:
      precision = precision_score(bb_preds, ruleMpreds)
    except:
      precision = None
    try:
      recall = recall_score(bb_preds, ruleMpreds)
    except:
      precision = None
    cm = confusion_matrix(bb_preds, ruleMpreds)
    
    return [acc, f1, precision, recall, cm]
                

def get_best_acc(scores_dict, clf_method, logger):
    vals = {0:'skope', 1:'rulefit', 2:'ripper'}
    accuracy_list = [scores_dict['%s_skope_accuracy' %(clf_method)], scores_dict['%s_rulefit_accuracy' %(clf_method)], scores_dict['%s_ripper_accuracy' %(clf_method)]]
    best_accuracy = max(accuracy_list)
    best_method_name = vals[accuracy_list.index(max(accuracy_list))]       
    
    if best_method_name == 'skope' and len(list(scores_dict['%s_skope_numpredictedClasses' %(clf_method)].keys())) == 1:
          best_accuracy = max(accuracy_list[1:])
          best_method_name = vals[accuracy_list.index(max(accuracy_list[1:]))]
    
    scores_dict['best_accuracy_%s' %clf_method] = best_accuracy
    scores_dict['best_method_%s' %clf_method] = best_method_name
    logger.info('best_method_%s is %s' %(clf_method, scores_dict['best_method_%s' %clf_method]))
    return scores_dict
            
                            
def compare_evaluate_plot(folder, elias, file_name, results_df, dm, logger):
    scores_dict = {}
    y_labels = results_df['actual']
    for clf_method in ['xgboost', 'logit', 'gbm']:
        bb_preds = results_df['preds_%s'% clf_method]
        scores_dict['accuracy_%s_actual' %clf_method] = accuracy_score(y_labels, bb_preds)
        accuracy_list = []
        
        for rule_method in ['skope', 'ripper', 'rulefit']:
            ruleMpreds = results_df['preds_%s'% rule_method]
            scores_dict['accuracy_%s_actual' %rule_method] = accuracy_score(y_labels, ruleMpreds)
            scores_dict['%s_%s_numpredictedClasses' %(clf_method, rule_method)] = Counter(ruleMpreds)
            metrics_list = ['accuracy', 'f1_score', 'precision', 'recall', 'confusion_matrix']
            scores = evaluate(bb_preds, ruleMpreds) 

            for i in range(len(metrics_list)):
                scores_dict['%s_%s_%s' %(clf_method, rule_method, metrics_list[i])] = scores[i]
            fig, ax = plot_confusion_matrix(conf_mat=scores[4], colorbar=True, class_names=[dm.pos_label, dm.neg_label])
            plt.savefig(os.path.join(folder,'confusionMatrix_%s_%s_%s.png'%(clf_method, rule_method, file_name)), dpi=300, bbox_inches='tight')

        

    scores_dict = get_best_acc(scores_dict, clf_method, logger)
    save_pkl(folder, 'scores_clfVSrulesTest', file_name, scores_dict)
    return    
   

def match_vals(row):
    if row[0] == row [1]:
        return 1
    else:
        return 0
                            
                            
datasets = ['sepsis2','production','traffic_fines', 'BPIC2017_O_Accepted']
for method_name in ['single_agg', 'prefix_index']: 
    bucket_method , cls_encoding = method_name.split('_')
    methods = encoding_dict[cls_encoding]
    
    for dataset_name in datasets:
        logger_name = os.path.join(outputs, f"RuleBasedPredLog_%s_%s.log" % (dataset_name, method_name))
        logger = get_logger(logger_name)
        #logging.basicConfig(filename=os.path.join(outputs, "executionLog_%s_%s.log" %(dataset_name, method_name)), format='%(asctime)s %(message)s', filemode='a') 
        
    
    
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
        bucketer_args = {'encoding_method': "agg",
                             'case_id_col': dm.case_id_col,
                             'cat_cols': [dm.activity_col],
                             'num_cols': [],
                             'random_state': 22}
    
        bucketer = get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(df_train_prefixes)
        bucket_assignment_test = bucketer.predict(df_test_prefixes)
        
     
        for bucket in set(bucket_assignment_test):
                relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
                relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
                test_case_ids = list(relevant_test_bucket.values)
                df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)

                df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
                train_y_experiment = dm.get_label_numeric(df_train_bucket)
                prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]   
                
                featureCombinerExperiment = FeatureUnion([(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])
                encoded_training = featureCombinerExperiment.fit_transform(df_train_bucket)
                ffeatures = featureCombinerExperiment.get_feature_names()
                training_set_df = pd.DataFrame(encoded_training, columns=ffeatures)
                  
                train_bkt_size = training_set_df.shape[0]
                feat_num = training_set_df.shape[1]                        
                
               
                train_file_name = '%s_%s_%s_%s_%s' % (dataset_name, method_name, train_bkt_size, prfx_len, feat_num)
                logger.info('now working on: {0}'.format(train_file_name))
                test_buckets_grouped = df_test_bucket.groupby(dm.case_id_col)
                encoded_testing = featureCombinerExperiment.fit_transform(df_test_bucket)
                testing_set_df = pd.DataFrame(encoded_testing, columns=ffeatures)
                test_y_real = dm.get_label_numeric(df_test_bucket)
                
                logger.info('vals in test_y_real: {0}'.format(Counter(test_y_real)))
                test_bkt_size = testing_set_df.shape[0]
                test_file_name = '%s_%s_%s_%s_%s' % (dataset_name, method_name, test_bkt_size, prfx_len, feat_num)
               
                discretized_test_df_file = open(os.path.join(discretized_logs, 'discretizedDF_Testing_%s.pkl' %test_file_name),'rb')
                discretized_test_df = pickle.load(discretized_test_df_file)
                
                results_df = pd.DataFrame({"case_ids": test_case_ids, "actual": test_y_real})
                for clf_method in ['logit', 'gbm', 'xgboost']:
                    preds_file = open(os.path.join(bb_preds, 'predictions_%s_%s.pkl'%(clf_method, train_file_name)), 'rb')
                    preds_bb = pickle.load(preds_file)
                    
                    clf_preds_col = 'preds_%s'%clf_method
                    results_df[clf_preds_col] = preds_bb
                    
                  
                for rules_method in ['skope', 'ripper', 'rulefit']:
                    logger.info('now working on: {0} and the rule-based classifier is {1}'.format(train_file_name, rules_method))
                    
                    rules_input_dir = os.path.join(rules_dir, rules_method)

                    rule_preds = []
                    
                    rule_model_file = open(os.path.join(rules_input_dir,'%sModel_%s_training.pkl'%(rules_method.capitalize(), train_file_name)), 'rb')
                    rule_model = pickle.load(rule_model_file)
                    
                    start = time.time()
                    if rules_method == 'skope':
                        
                        rule_preds = rule_model.predict_top_rules(encoded_testing, len(rule_model.rules_))
                        
                        logger.info('for this event log, classes of predictions with skope are: {0}'.format(list(set(rule_preds))))
                        logger.info('for this event log, number of skope rules is: {0}'.format(len(rule_model.rules_)))
                    
                    elif rules_method == 'rulefit':
                        
                        rulefit_preds = rule_model.predict(encoded_testing)
                        rule_preds = [1 if label == dm.pos_label else 0 for label in rulefit_preds]
                        rulefit_preds = rule_preds
                        
                    elif rules_method == 'ripper':
                        
                        ripper_preds = rule_model.predict(testing_set_df)
                        rule_preds = [1 if label == dm.pos_label else 0 for label in ripper_preds]

                    end_pred = time.time() - start
                   
                    
                    logger.info('got all the predictions for the test set: {0} in time {1}'.format(test_file_name, end_pred))
                    
                    
                    rule_preds_col = 'preds_%s'%rules_method
                    results_df[rule_preds_col] = rule_preds
                    
                    for col in ['actual', 'preds_xgboost', 'preds_logit', 'preds_gbm']:
                      match_label_col = 'match_%sVs%s' %(rules_method,col)
                      results_df[match_label_col] = results_df[[col, rule_preds_col]].apply(match_vals, axis=1)
 
                    del rule_preds, rule_model
                
                  
                    
                save_pkl(outputs, 'predictions_clfVSrulesTest', train_file_name, results_df)   
                compare_evaluate_plot(outputs, 'evaluations_plots', train_file_name, results_df, dm, logger)      
                
  

            
    
            
             
    
            
                              
                       