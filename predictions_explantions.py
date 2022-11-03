import xgboost as xgb
import pandas as pd #using pandas==0.25
import numpy as np
import time
import sys
import os
import re
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
import pickle
from collections import defaultdict

import io
import shutil
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from rulefit import RuleFit

from lime_stability.stability import LimeTabularExplainerOvr
import shap
from wittgenstein.ripper import RIPPER
from wittgenstein.interpret import interpret_model, score_fidelity
#from aix360.algorithms.rbm import FeatureBinarizer, LogisticRuleRegression

import warnings
warnings.simplefilter('ignore')
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()


from helpers.DatasetManager import DatasetManager
from helpers.Bucketers import get_bucketer
from helpers.Encoders import get_encoder
from utils.retrieval import retrieve_artefact
import Definitions

#Defining Basic parameters
n_iter = 5
datasets = ['sepsis2',  'traffic_fines', 'production',  "BPIC2017_O_Accepted"]
#cls_methods = ['logit', 'rf', 'xgboost', 'rulefit', 'mlp']
cls_methods = ['logit', 'rf', 'xgboost',  'mlp']
method_names = ['single_agg', 'prefix_index']


bucket_encoding = "agg"
encoding_dict = { "agg": ["static", "agg"], "index": ["static", "index"]}

train_ratio = 0.8
n_splits = 3
random_state = 22

artefacts_dir = 'localXAIEval_October22' 
params_dir = os.path.join(artefacts_dir,'cv_results_Oct22')
saved_artefacts = os.path.join(artefacts_dir, 'models_artefacts')
shap_dir = os.path.join(artefacts_dir,'shap_artefacts')
lime_dir = os.path.join(artefacts_dir,'lime_artefacts')
rules_dir = os.path.join(artefacts_dir,'rules')



def get_variants_df(variants_file_name, data, dm, max_prfx_len):
    """
    a function constructing a dataframe containing case_ids, variant represented by each case and the percentage of instances representing 
    this variant in the event log 
    parameters:
    - folder: input folder
    - variants_file_name: output file name
    - data: complete event log
    - dm: object from the data manager class
    - max_prfx_len : maximum prefix length for this event log
    """
    variants_df = pd.DataFrame(columns=[dm.case_id_col,'variant'])
    variants_df[dm.case_id_col] = data.sort_values(dm.timestamp_col, kind="mergesort").groupby(dm.case_id_col).groups.keys()
    variants_df['variant'] = data.sort_values(dm.timestamp_col, kind="mergesort").groupby(dm.case_id_col).head(max_prfx_len).groupby(dm.case_id_col, as_index=False)[dm.activity_col].apply(lambda x: "__".join(list(x)))
    #percentage of the instances representing the variant in each case
    variants_df['Percentage'] = variants_df.groupby(['variant'])[dm.case_id_col].transform(len)/variants_df.shape[0]
    variants_df.to_csv(variants_file_name,sep=';', index=False)
    return variants_df


def get_samples(folder, variants_file_name, variants_df, dm, original_data, n, flag):
    """
    a function for sampling an event log based on variants percentages
    parameters:
    - folder: folder containing the variants file
    - variants_file_name
    - variants_df: a dataframe containing case_ids along with their relevant variants and percentages
    - dm: object from the data manager class
    - original_data : used to get instances selected to be in the sample event log
    - n : number of samples to be obtained
    - flag: an indicator of the type of the event log to be sampled
    outputs:
    - selected_instances: case_ids of instances selected to be in the sample event log
    - sampled_df_bucket: instances of the selected samples obtained from the original event log
    - sampled_y: values of the target class from the original event log
    """
    args = {'random_state' : 1, 'weights' : 'Percentage'}
    #to sample 100 instances from each event log
    if variants_df.shape[0] < 100:
       if flag == 'training':
         args['n'] = variants_df.shape[0]
       elif variants_df.shape[0] > 20: 
         args['n'] = n
       else:
         args['n'] = variants_df.shape[0]
    else:
       args['n'] = n
    sampled_set = variants_df.sample(**args)
    selected_instances = sampled_set[dm.case_id_col].tolist()
    with open(os.path.join(folder, variants_file_name), 'a') as fout:
                fout.write('\n')
                fout.write('%s;%s\n' %('Selected instances:', selected_instances))             
    sampled_df_bucket = original_data[original_data[dm.case_id_col].isin(selected_instances)]
    selected_samples_indices = original_data.index[original_data[dm.case_id_col].isin(selected_instances)].tolist()
    sampled_y = dm.get_label_numeric(sampled_df_bucket)
    prfx_len_sample_df = dm.get_prefix_lengths(sampled_df_bucket)[0]
    return selected_instances, sampled_df_bucket,sampled_y, selected_samples_indices


def make_exp_file(lime_dir, file_name, flag):
    """
    a function to construct a file and write its header
    parameters:
    - folder: output folder
    - file_name: event log information
    - flag: to indicate the type of event log which was sampled
    """
    resexpfile = os.path.join(lime_dir, 'LIMEexplanations_SampledDF_%s_%s.csv' %(flag,file_name))
    with open(resexpfile, 'w+') as resf:
        header2 = ['Case_ID', 'Explanation', 'RealClass', 'PredictedClass']
        writer2 = csv.DictWriter(resf, delimiter=';', lineterminator='\n', fieldnames=header2)
        writer2.writeheader()
    return resexpfile


def visualise_lime_exp(lime_dir, elias, exportedexp, class_names, figsize=(4,4)):
    """
    a function to plot selected instance from the sampledDF as a barplot
    parameters: 
    -folder: to save outputs
    -figfile_name: string to be used in formatting the output figure
    -exportedexp: the explanation as a list of tuples
    -figsize: size of the returned plot
    """
    fig = plt.figure(figsize=figsize)
    vals = [x[1] for x in exportedexp[0:11]]
    names = [x[0] for x in exportedexp[0:11]]
    vals.reverse()
    names.reverse()
    colors = ['green' if x > 0 else 'red' for x in vals]
    pos = np.arange(len(vals)) + .5
    plt.barh(pos, vals, align='center', color=colors)
    plt.yticks(pos, names)
    title = 'Local explanation for class %s' %class_names[1]
    plt.title(title)
    plt.savefig(os.path.join(lime_dir,'LimeExpTopfeats_%s.png'%elias), dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    return


def visualise_shap_exp(shap_dir,elias, shap_vals, expected_val, idx, ffeatures, case_locator, encoded_testing):
    """
    a function to plot selected instance from the sampledDF as a waterfall plot
    parameters: 
    -folder: to save outputs
    -figfile_name: string to be used in formatting the output figure
    -shap_vals: shap values of the whole explained dataset
    -expected_val: expected value obtained from the shap explainer
    -case_id: of the instance to be plotted
    """        
    shap_idx = [x for (x, y, z) in case_locator if y == idx][0]
    #shap.plots._waterfall.waterfall_legacy(expected_val, shap_vals[shap_idx], feature_names=ffeatures)
    shap.force_plot(expected_val, shap_vals[shap_idx], show=False, matplotlib=True)
    plt.savefig(os.path.join(shap_dir,'SHAPforce_%s.png'%elias), dpi=300, bbox_inches='tight');
    shap.decision_plot(expected_val, shap_vals[shap_idx], encoded_testing[shap_idx],
                       feature_names=ffeatures, show=False)
    plt.savefig(os.path.join(shap_dir, 'SHAPdecision_%s.png'%elias), dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    return


def visualise(elias, exportedexp, class_names, expected_val, idx, args,  specific_args):
    visualise_lime_exp(args['lime_dir'], elias, exportedexp, class_names, (4,4))
    visualise_shap_exp(args['shap_dir'], elias, specific_args['vals'], expected_val, idx, args['ffeatures'], specific_args['case_locator'], specific_args['encoded_data'])


def explain_single_samples(limeexplainer, expected_val,class_names, args, specific_args):
    resexpfile_testing = make_exp_file(args['lime_dir'], args['file_name_complete'], specific_args['flag']) 
    count = 0  
    explained_samples, minority_samples = [], [] 
    groupped_testsamples = specific_args['sample_testing_bucket'].groupby(args['dm'].case_id_col)
        
    for idx, instance in groupped_testsamples:
            count+= 1          
            real_y = args['dm'].get_label_numeric(instance)         
            case_id = args['dm'].get_case_ids(instance)          
            print(real_y)
            encoded_testing_group = args['transformer'].fit_transform(instance)
            print('done here')
            test_instance = np.transpose(encoded_testing_group[0])   
            print('done here, too')
            predicted_y = args['model'].predict(encoded_testing_group) 
            print(predicted_y)         
            if predicted_y == real_y:        #to explain only TN or TP predictions in the sampled_testset           
                expparams = {"data_row": test_instance, "predict_fn":args['model'].predict_proba,
                                "num_features": len(args['ffeatures']), "distance_metric": "euclidean", \
                             "labels": (args['minority_numeric'],)}
                explanation = limeexplainer.explain_instance(**expparams)
                exportedexp = explanation.as_list()
                explained_samples.append((case_id, real_y))
                if count %10 == 0:
                    elias = '%s_%s_%s' %(specific_args['flag'], args['file_name_complete'],case_id)
                    visualise(args['lime_dir'], args['shap_dir'], elias, exportedexp, class_names, expected_val, idx, args, specific_args)          
                with open(resexpfile_testing, 'a') as resf:
                    resf.write('%s;%s;%s;%s\n' % (case_id, exportedexp, real_y, predicted_y))
                
                if real_y[0] == args['minority_numeric']:
                  minority_samples.append(case_id)
    return explained_samples, minority_samples


def transform_shap_vals(shap_dir, file_name_complete, shap_values_df, model, selected_instances, sampled_y, encoded_sample, flag):
    transformed_shapvals_csv = os.path.join(shap_dir, 'TransformedShapValues_SampledDF_%s_%s.csv' %(flag, file_name_complete))
    shap_values_df = shap_values_df.abs()
    print('passed this')
    shap_values_df['Case_ID'] = selected_instances 
    shap_values_df['real_y'] = sampled_y
    print('okay')
    shap_values_df['predictions'] = model.predict(encoded_sample) 
    print('everything is fine')
    shap_values_df.to_csv(transformed_shapvals_csv, sep=';', index=False)
    return 


def shap_values_demonstrate(args,specific_args): 
    shapvals_csv = os.path.join(args['shap_dir'], 'ShapValues_SampledDF_%s_%s.csv' %(specific_args['flag'], args['file_name_complete']))
    shap_values_df = pd.DataFrame(specific_args['vals'], columns = args['ffeatures'])
    print('shap_vals_df: %s,%s' %shap_values_df.shape)
    shap_values_df.to_csv(os.path.join(args['shap_dir'], shapvals_csv), sep=';', index=False)
    print('Here now')
    transform_shap_vals(args['shap_dir'], args['file_name_complete'], shap_values_df, args['model'], specific_args['selected_instances'], specific_args['sampled_y'], specific_args['encoded_sample'], specific_args['flag']) 
    shap.summary_plot(specific_args['vals'], specific_args['encoded_data'], feature_names=args['ffeatures'], plot_type='bar', show=False, max_display=10)
    plt.savefig(os.path.join(args['shap_dir'], 'Shap_values_barSampledTrainDF_%s.png' %args['file_name_complete']), dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()
    return 


      

# final experiments
saved_artefacts = os.path.join(artefacts_dir, 'model_and_hdf5')
if not os.path.exists(saved_artefacts):
    os.makedirs(os.path.join(saved_artefacts))

results_dir_final = os.path.join(artefacts_dir, 'final_experiments_results')
if not os.path.exists(results_dir_final):
    os.makedirs(os.path.join(results_dir_final))


for dataset_name in datasets:
    for cls_method in cls_methods:
       for method_name in method_names:
          print('%s_%s_%s'%(dataset_name, cls_method, method_name))
          if method_name == "single_agg":
            gap = 1
          else:
            gap = 5
          bucket_method , cls_encoding = method_name.split('_')
          methods = encoding_dict[cls_encoding]
          params_file = os.path.join(params_dir, 'optimal_params_%s_%s_%s.pickle' % (cls_method, dataset_name, method_name))
          with open(params_file, 'rb') as fin:
              args = pickle.load(fin)
      
          current_args = {}
      
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
      
          if gap > 1:
              outfile = os.path.join(results_dir_final, "performance_experiments_%s_%s_%s_gap%s.csv" % (
              cls_method, dataset_name, method_name, gap))
              
          else:
              outfile = os.path.join(results_dir_final,
                                     'performance_experiments_%s_%s_%s.csv' % (cls_method, dataset_name, method_name))
              
      
          
          if (method_name == 'prefix_index'):
              df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final, gap=5)
          else:
              df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final)
          
      
          for i in range(n_iter):
              print('starting Iteration number {0} in file {1}'.format(i, dataset_name))
              if (method_name == 'prefix_index'):
                  df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final, gap=5)
              else:
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
      
              preds_all = []
              test_y_all = []
              nr_events_all = []
              
      
              for bucket in set(bucket_assignment_test):
                  print('%s_%s' %(bucket_method, bucket))
                  current_args = args
      
                  relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
                  relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
                  df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)
                  test_prfx_len = dm.get_prefix_lengths(df_test_bucket)[0]
                  nr_events_all.extend(list(dm.get_prefix_lengths(df_test_bucket)))
                  if len(relevant_train_bucket) == 0:
                      preds = [dm.get_class_ratio(train)] * len(relevant_test_bucket)
                      
                  else:
                      df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
                      train_y_experiment = dm.get_label_numeric(df_train_bucket)
                      prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]
                      pos_class_ratio = df_train_bucket.groupby(dm.case_id_col, as_index=False).first()[dm.label_col].value_counts()[dm.pos_label]/ len(df[dm.case_id_col].unique())
                      neg_class_ratio = df_train_bucket.groupby(dm.case_id_col, as_index=False).first()[dm.label_col].value_counts()[dm.neg_label]/ len(df[dm.case_id_col].unique())
                      if neg_class_ratio < pos_class_ratio:
                          minority_class, minority_numeric = dm.neg_label, 0
                      else:
                          minority_class, minority_numeric = dm.pos_label, 1
                          
                      if len(set(train_y_experiment)) < 2:
                          preds = [train_y_experiment[0]] * len(relevant_train_bucket)
                          test_y_all.extend(dm.get_label_numeric(df_test_prefixes))
                      else:
                          featureCombinerExperiment = FeatureUnion(
                              [(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])
      
                          if cls_method == 'xgboost':
                              cls_experiment = xgb.XGBClassifier(objective='binary:logistic',
                                                                 n_estimators=500,
                                                                 learning_rate=current_args['learning_rate'],
                                                                 max_depth=current_args['max_depth'],
                                                                 subsample=current_args['subsample'],
                                                                 colsample_bytree=current_args['colsample_bytree'],
                                                                 min_child_weight=current_args['min_child_weight'],
                                                                 seed=random_state)              
                          elif cls_method == 'logit':
                              cls_experiment = LogisticRegression(C=2 ** current_args['C'], random_state=random_state)
                          elif cls_method == 'rf':
                              cls_experiment = RandomForestClassifier(n_estimators=500, max_features=current_args['max_features'],
                                                           random_state=random_state)
                          elif cls_method == 'mlp':
                              input_shape = df_train_bucket.shape
                              size1 = (input_shape[1]+2)//2
                              size2 = int(input_shape[0]/(2*(input_shape[1]+2)))
                              cls_experiment = MLPClassifier(hidden_layer_sizes= (size2,), 
                                                             max_iter = current_args['max_iter'], activation = current_args['activation'], 
                                                             solver = current_args['solver'], alpha=current_args['alpha'],
                                                             learning_rate=current_args['learning_rate']) #current_args['hidden_layer_sizes']
                          elif cls_method == 'rulefit':
                            cls_experiment = RuleFit(tree_size=current_args['tree_size'], sample_fract='default', max_rules=2000, memory_par=0.01,
                                              tree_generator=None, rfmode='classify', lin_trim_quantile=current_args['lin_trim_quantile'], 
                                              lin_standardise=True, exp_rand_tree_size=True, random_state=1)    
                          
                                                                     
      
                          if cls_method in ["logit", 'mlp']:
                              pipeline_final = Pipeline([('encoder', featureCombinerExperiment), ('scaler', StandardScaler()),
                                                         ('cls', cls_experiment)])
                          elif cls_method in ['rf', 'xgboost', 'rulefit']:
                              pipeline_final = Pipeline([('encoder', featureCombinerExperiment), ('cls', cls_experiment)])
                          
      
                          
                          pipeline_final.fit(df_train_bucket, train_y_experiment)
                          
                          
                          
                          ffeatures = pipeline_final.named_steps['encoder'].get_feature_names()
                          encoded_training = featureCombinerExperiment.fit_transform(df_train_bucket)
                          
                          
                          
                          encoded_testing = featureCombinerExperiment.fit_transform(df_test_bucket)   
                          train_bucket = get_variants_df('variants_and_weights_ofEventLog_training_%s.csv' %dataset_name, 
                                                         df_train_bucket, dm, max_prefix_length_final)                                   
                          test_bucket = get_variants_df('variants_and_weights_ofEventLog_testing_%s.csv' %dataset_name, 
                                                        df_test_bucket, dm, max_prefix_length_final)
                          
                          if i == n_iter-1:
                              training_set_df = pd.DataFrame(encoded_training, columns=ffeatures)
                              prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]
                              bkt_size = training_set_df.shape[0]
                              feat_num = training_set_df.shape[1]
                              cls = pipeline_final.named_steps['cls'] 
                              # save the model for later use
                              model_file = os.path.join(saved_artefacts, 'model_%s_%s_%s_%s_%s_%s.pickle' % (
                              cls_method, dataset_name, method_name, bkt_size, prfx_len, feat_num))
                              with open(model_file, 'wb') as fout:
                                  pickle.dump(cls, fout)
                                  
                              #construct sample dataframes    
                              selected_instances_training, sampled_df_train_bucket,sampled_train_y, selected_samples_training_indices= \
                                      get_samples(artefacts_dir,'variants_and_weights_ofEventLog_training_%s.csv' %dataset_name, 
                                                  train_bucket, dm, df_train_bucket,500, 'training')      
                              selected_instances_testing, sampled_df_test_bucket,sampled_test_y, selected_samples_testing_indices= \
                                      get_samples(artefacts_dir,'variants_and_weights_ofEventLog_testing_%s.csv' %dataset_name, 
                                                  test_bucket, dm, df_test_bucket,100, 'testing')
                              
                              """encoded_training_sample = featureCombinerExperiment.transform(sampled_df_train_bucket)
                              encoded_training_sample_df = pd.DataFrame(encoded_training_sample, columns=ffeatures)
                              case_locator_training = list(zip(encoded_training_sample_df.index.tolist(), 
                                                               selected_instances_training, selected_samples_training_indices))"""
                              
                              encoded_testing_sample = featureCombinerExperiment.transform(sampled_df_test_bucket)
                              encoded_testing_sample_df = pd.DataFrame(encoded_testing_sample, columns=ffeatures)                        
                              case_locator_testing = list(zip(encoded_testing_sample_df.index.tolist(), 
                                                              selected_instances_testing, selected_samples_testing_indices))
                              
                              class_names = [dm.neg_label, dm.pos_label]
                              # using wrapped module from Lime_stability libarary
                              limeexplainer = LimeTabularExplainerOvr(encoded_training, mode='classification',
                                                                                        feature_names=ffeatures,
                                                                                        class_names=[dm.neg_label, dm.pos_label],
                                                                                        discretize_continuous=True,
                                                                                        verbose=True)
                              exp_dirs = {}
                              file_name_complete = '%s_%s' %(cls_method, dataset_name)                            
                              for elias in ['lime', 'shap']:
                                      exp_dirs[elias] = '%s_%s' %(elias, file_name_complete)
                                      if not os.path.exists( exp_dirs[elias]):
                                          os.makedirs( exp_dirs[elias])                                                              
                                                                              
                              if cls_method in ['xgboost', 'rf']:
                                      explainer = shap.TreeExplainer(cls, feature_names=ffeatures)                          
                                      if cls_method == 'rf':
                                          shap_values_testing = explainer.shap_values(encoded_testing_sample, check_additivity=False)[minority_numeric]
                                      else:
                                          shap_values_testing = explainer.shap_values(encoded_testing_sample, check_additivity=False)
                              elif cls_method in ['logit', 'rulefit']:
                                      explainer = shap.LinearExplainer(cls,encoded_training)
                                      shap_values_testing = explainer.shap_values(encoded_testing_sample)     
                              elif cls_method == 'mlp': 
                                      explainer = shap.SamplingExplainer(cls.predict_proba,encoded_training, feature_names=ffeatures) 
                                      shap_values_testing = explainer.shap_values(encoded_testing_sample)[minority_numeric]                                                       
                              
                            
                              
                              if cls_method in ['rf', 'mlp']:
                                  expected_value = explainer.expected_value[minority_numeric]
                              else:
                                  expected_value = explainer.expected_value
                              out1 = os.path.join( 'shap_explainer_%s.pickle' %file_name_complete)
                              with open(out1, 'wb') as output:
                                    pickle.dump(explainer, output)
                              
                              test_y_experiment = dm.get_label_numeric(df_test_bucket)
                              args = { 'file_name_complete':file_name_complete, 'dm':dm, 'ffeatures':ffeatures, 'model':cls, \
                                  'lime_elias': 'LIMEexplanations_SampledDF', 'shap_elias': 'TransformedShapValues_SampledDF',\
                                  'minority_numeric':minority_numeric, 'transformer':featureCombinerExperiment,\
                                   'training_encoded':encoded_training, 'train_y':train_y_experiment, 'shap_dir':shap_dir, \
                                   'lime_dir': lime_dir}                                       
                              args_testing = {'vals':shap_values_testing, 'selected_instances':selected_instances_testing,\
                                  'sampled_y':sampled_test_y, 'encoded_sample':encoded_testing_sample, 'encoded_data':encoded_testing,\
                                    'flag':'testing', 'case_locator':case_locator_testing, \
                                    'encoded_sample_df': encoded_testing_sample_df, 'sample_testing_bucket': sampled_df_test_bucket,\
                                    'test_y_experiment':test_y_experiment}
                                                          
                              shap_values_demonstrate(args, args_testing)
                              explained_samples,minority_samples = explain_single_samples(limeexplainer, expected_value,class_names, args, args_testing)
                              
                              with open(os.path.join(results_dir_final, "explainedInstances.csv"), 'a') as out:                             
                                  out.write('%s;%s;%s;%s\n' % (cls_method, dataset_name, method_name, gap))
                                  out.write('explained_samples; %s\n' % (explained_samples))
                                  out.write('minority_samples; %s\n' % (minority_samples))                                                              
                          
                          
                          with open(outfile, 'a') as out:
                              out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                              'dataset', 'method', 'cls', 'nr_events', 'n_iter', 'prefix_length', 'score'))
                              out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                              dataset_name, method_name, cls_method, -1, i, prfx_len, -1))
      
                          preds = []
                          test_y_bucket = []
                          test_buckets_grouped = df_test_bucket.groupby(dm.case_id_col)                  
      
                          for idx, grouppred in test_buckets_grouped:                        
                              test_y_all.extend(dm.get_label_numeric(grouppred))
                              if method_name == 'prefix_index':
                                  test_y_bucket.extend(dm.get_label_numeric(grouppred))
                              if cls_method == 'rulefit':
                                  pred = pipeline_final.predict(grouppred)
                              else:
                                  preds_pos_label_idx = np.where(cls_experiment.classes_ == 1)[0][0]
                                  pred = pipeline_final.predict_proba(grouppred)[:, preds_pos_label_idx]                   
                              preds.extend(pred)
                                                                                 
                  preds_all.extend(preds)
      
      
          with open(outfile, 'w') as out:
              out.write('%s;%s;%s;%s;%s;%s;%s\n' % ('dataset', 'method', 'cls', 'nr_events', 'n_iter', 'metric', 'score'))
              df_results = pd.DataFrame({'actual': test_y_all, 'predicted': preds_all, 'nr_events': nr_events_all})
              for nr_events, group in df_results.groupby('nr_events'):
                  if len(set(group.actual)) < 2:
                      out.write(
                          "%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc", np.nan))
                  else:
                      try:
                          out.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc",
                                                                roc_auc_score(group.actual, group.predicted)))                    
                      except ValueError:
                          pass
              try:
                  out.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "auc",
                                                        roc_auc_score(df_results.actual, df_results.predicted)))           
              except ValueError:
                  pass
      
             
      
