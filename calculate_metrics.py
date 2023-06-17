import os
import math
import pickle
import fnmatch
import pandas as pd
#from rbo import rbo
import re
import sys
import numpy as np
import logging
from sklearn.metrics import accuracy_score
from helpers.DatasetManager import DatasetManager
from helpers.utils import  save_pkl, get_logger

out_dir = 'artefacts'
in_dir = 'rules'


    
    
def clean_name(x, elias):
    if re.search('.pkl', x):
          x = x.replace('.pkl','')
    for o in ['Rows_', elias]:
       if re.search(o, x):
          x = x.partition(x)[-1]
      
    return x
    

def compute_jaccard_sim(row):  
    try:
      return len(set(row[0]) & set(row[1])) / len(set(row[0]) | set(row[1]))
    except:
      return 0
        
def compute_coverage(row):
    if isinstance(row[0], int) and isinstance(row[1], int):
      return row[0] / row[1]
    else:
      return 0
        
        
        
def rbo(S,T, p= 0.9):
    # source: https://towardsdatascience.com/how-to-objectively-compare-two-ranked-lists-in-python-b3d74e236f6a
    """ Takes two lists S and T of any lengths and gives out the RBO Score
    Parameters
    ----------
    S, T : Lists (str, integers)
    p : Weight parameter, giving the influence of the first d
        elements on the final score. p<0<1. Default 0.9 give the top 10 
        elements 86% of the contribution in the final score.
    
    Returns
    -------
    Float of RBO score
    """
    
    # Fixed Terms
    k = max(len(S), len(T))
    x_k = len(set(S).intersection(set(T)))
    
    summation_term = 0

    # Loop for summation
    # k+1 for the loop to reach the last element (at k) in the bigger list    
    for d in range (1, k+1): 
            # Create sets from the lists
            set1 = set(S[:d]) if d < len(S) else set(S)
            set2 = set(T[:d]) if d < len(T) else set(T)
            
            # Intersection at depth d
            x_d = len(set1.intersection(set2))

            # Agreement at depth d
            a_d = x_d/d   
            
            # Summation
            summation_term = summation_term + math.pow(p, d) * a_d

    # Rank Biased Overlap - extrapolated
    rbo_ext = (x_k/k) * math.pow(p, k) + ((1-p)/p * summation_term)

    return rbo_ext
    
              
def get_cols(flag):
    
    if flag in ['anchor', 'lore']:
        cols = ['row_idx', '%s_LenExpRule' %flag ,  '%s_expRuleFeats' %flag,  'match_%sVsactual' %flag, 'match_%sVspreds_xgboost' %flag,\
      'match_%sVspreds_logit' %flag, 'match_%sVspreds_gbm' %flag, 'coverage_vol_%s' %flag]
      

    
    elif flag == 'global':
        Mcols = ['%s_%s'%(x,y) for x in  ['skope', 'ripper', 'rulefit']  for y in ['ruleLen', 'ruleCoverage', 'rule_expRuleFeats']]
        Predcols = ['match_%sVs%s' %(x,y) for x in ['skope', 'ripper', 'rulefit'] for y in ['actual', 'preds_xgboost', 'preds_gbm', 'preds_logit'] ] 
        cols = ['case_ids'] + Mcols + Predcols
    
    return cols



def get_df(in_dir, fname, xai_method, clf_method):
      elias = 'prefinal_%sExp_' %xai_method
      #xai_measuresfile = elias + fname
      xai_dir = in_dir
      
      
      for exp_f in os.listdir(xai_dir):
        
          if all([x in exp_f for x in [fname, elias, clf_method]]):
              #exp_measures_file = open(os.path.join(xai_dir,xai_measuresfile+'.pkl'), 'rb') 
              exp_measures_file = open(os.path.join(xai_dir,exp_f), 'rb') 
              xai_df = pickle.load(exp_measures_file)
              xai_df.rename(columns={'coverage_vol': 'coverage_vol_%s' %xai_method}, inplace=True)
              retrieved_cols = get_cols(xai_method)
                            

                 
              xai_measurements_data =  xai_df[retrieved_cols]
          else:
              continue
      
          return xai_measurements_data
           


def get_experimental_cases(lore_df, anchor_df, measurements_data, logger):
    
    intersection = pd.merge(lore_df, anchor_df, how='inner', on=['row_idx'])
    intersection.dropna(inplace=True)
    
    data_selected_instances = measurements_data.loc[measurements_data['case_ids'].isin(list(intersection['row_idx']))]
    
    logger.info('selected instances are: {0}'.format(list(intersection['row_idx'])))
    logger.info('shape of global data: {0}'.format(measurements_data.shape))
    logger.info('shape of anchor data: {0}'.format(anchor_df.shape))
    logger.info('shape of lore data: {0}'.format(lore_df.shape))
    logger.info('shape of global (selected) data: {0}'.format(data_selected_instances.shape))
    logger.info('shape of local (selected)  data: {0}'.format(intersection.shape))
        
    return data_selected_instances, intersection
    

def get_data(in_dir, fname, flag):
    for fname in os.listdir(in_dir):
          if fnmatch.fnmatch(fname, 'prefinal_%sExp_*%s.pkl' %(flag,name)):
              exp_measures_file = open(os.path.join(in_dir,fname), 'rb')
              xai_df = pickle.load(exp_measures_file)
              xai_df.rename(columns={'coverage_vol': 'coverage_vol_%s' %flag}, inplace=True)
              retrieved_cols = get_cols(flag)
                           
              if 'BPIC2017_O_Accepted_single_agg_494892_1_722' in fname:
                  
                  retrieved_cols.remove('match_%sVspreds_logit' %flag)
                 
              xai_measurements_data =  xai_df[retrieved_cols]
              print(xai_measurements_data.columns)
    return xai_measurements_data




def get_original_cols(fname):
    datasets = ['sepsis2', 'production', 'traffic_fines', 'BPIC2017_O_Accepted']
    for x in datasets:
        if x in fname:
          dm = DatasetManager(x)
          break
    cols = [dm.case_id_col] + dm.static_cat_cols + dm.dynamic_cat_cols + dm.static_num_cols + dm.dynamic_num_cols
    return cols


def transform_cols(feats_col, orig_cols):
    transformed_col = []
    for l in feats_col:
        new_l = []
        for c in orig_cols:
          for x in l:
            if c in x:
                l.remove(x)
                new_l.append(c)
        transformed_row = list(set(new_l))  
        transformed_col.append(transformed_row)
    return transformed_col 
    
    
all_files_results, all_sums = {}, {}
for fname in os.listdir(in_dir):
    
    fnames = []
    
    #replace with "Global_and_preds_selectedInstancesModified" when ready
    elias = 'Global_and_preds_selectedInstances_'
    if fnmatch.fnmatch(fname, '%s*.pkl' %elias):       


        name = fname.split(elias)[1].split('.pkl')[0]
        name = clean_name(name, elias)  
        clf_method = name.partition('_')[0]   
        data_file = open(os.path.join(in_dir,fname), 'rb')
        data = pickle.load(data_file)
         
        retrieved_cols = get_cols('global')
        measurements_data_all = data[retrieved_cols]
                
         
        logger_name = os.path.join(out_dir, f"Measurements_%s.log" %name)
        logger = get_logger(logger_name)
        #logger.info('working on : {0}'.format(name))
        
        lore_df = get_data(in_dir, name, 'lore')
        anchor_df = get_data(in_dir, name, 'anchor')
        
        
        measurements_data, local_df = get_experimental_cases(lore_df, anchor_df, measurements_data_all, logger)
        orig_cols = get_original_cols(fname)
        test_len = measurements_data_all.shape[0]
        
        final_results, file_sum = {}, {}
        cases_len = local_df.shape[0]
        print(fname)
        for local_method in ['lore', 'anchor']:
           lfeats_col = local_df['%s_expRuleFeats' %local_method]
           
           transformed_lfeats_col = transform_cols(lfeats_col.values, orig_cols)
           print(transformed_lfeats_col)
           
           coverage_series = local_df['coverage_vol_%s' %local_method].apply(lambda x: x/test_len)
           final_results['%s_coverage' %(local_method)] = np.mean(coverage_series)
           
           for global_method in  ['skope', 'ripper', 'rulefit']:
               gfeats_col = measurements_data['%s_rule_expRuleFeats' %global_method]
               transformed_gfeats_col = transform_cols(gfeats_col.values, orig_cols)
               
               jaccard_sim = list(map(compute_jaccard_sim, zip(transformed_lfeats_col, transformed_gfeats_col)))

               final_results['%s_%s_featsSim' %(local_method, global_method)] = np.mean(jaccard_sim)


               
               rbo_len = rbo(measurements_data['%s_ruleLen' %global_method].values, local_df['%s_LenExpRule' %local_method].values)
               final_results['%s_%s_len' %(local_method, global_method)] = rbo_len
               
               len_fractions_local = local_df['%s_LenExpRule' %local_method].apply(lambda x: x/len(orig_cols))
               final_results['%s_len_orig' %local_method] = len_fractions_local.mean()
               
               
               
               rbo_actual = rbo(measurements_data['match_%sVsactual' %(global_method)].values, local_df['match_%sVsactual' %(local_method)].values)
               final_results['%s_%s_actual' %(local_method, global_method)] = rbo_actual
               
               rbo_preds = rbo(measurements_data['match_%sVspreds_%s' %(global_method, clf_method)].values, local_df['match_%sVspreds_%s' %(local_method, clf_method)].values)
               final_results['%s_%s_%s' %(local_method, global_method, clf_method)] = rbo_preds
               
               

               
               file_sum[f'{clf_method}_{local_method}_{global_method}'] = final_results['%s_%s_featsSim' %(local_method, global_method)] + final_results['%s_%s_coverage' %(local_method, global_method)] +rbo_preds + final_results['%s_len_orig' %local_method]
               print('----------------------------')
        all_files_results[name] = final_results
        all_sums[name] = file_sum
        print('*******************************************************************************************************************************')     

save_pkl(out_dir, 'all_measurements', 'final', all_files_results)
save_pkl(out_dir, 'all_sums', 'final', all_sums)     
scores_df = pd.DataFrame.from_dict(all_sums, orient='index')
scores_df.to_csv(os.path.join(out_dir, 'final_scores.csv'), sep=';')       
all_files_results_df = pd.DataFrame.from_dict(all_files_results, orient='index')
all_files_results_df.to_csv(os.path.join(out_dir, 'final_metrics.csv'), sep=';')
        
        
        
        
        