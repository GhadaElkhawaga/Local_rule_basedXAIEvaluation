import pandas as pd
import os
import csv
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion
from helpers.DatasetManager import DatasetManager
from helpers.Bucketers import get_bucketer
from helpers.Encoders import get_encoder
from utils.retrieval import retrieve_artefact


saved_artefacts = os.path.join('model_and_hdf5')
logs = os.path.join('logs')
variants_folder =os.path.join('Variants')
if not os.path.exists(variants_folder):
    os.makedirs(variants_folder)
encoding_dict = {
    'agg' : ['static', 'agg'],
    'index' : ['static', 'index']}


def get_variants_df(folder, variants_file_name, data, dm, max_prfx_len):
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
    variants_df.to_csv(os.path.join(folder, variants_file_name),sep=';', index=False)
    return variants_df
    
    
datasets = ['sepsis2', 'production' , 'traffic_fines', 'BPIC2017_O_Accepted']
for method_name in ['prefix_index','single_agg']: 
    bucket_method , cls_encoding = method_name.split('_')
    methods = encoding_dict[cls_encoding]
    for ds in datasets:
      dm = DatasetManager(ds)
      df = dm.read_dataset()
      cls_encoder_args_final = {'case_id_col': dm.case_id_col,
                              'static_cat_cols': dm.static_cat_cols,
                              'dynamic_cat_cols': dm.dynamic_cat_cols,
                              'static_num_cols': dm.static_num_cols,
                              'dynamic_num_cols': dm.dynamic_num_cols,
                              'fillna': True}        
      min_prefix_length_final = 1
      if "traffic_fines" in ds:
          max_prefix_length_final = 10
      elif "BPIC2017" in ds:
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
              featureCombinerExperiment = FeatureUnion(
                        [(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])
              relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
              relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
              df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
              df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)
              prfx_len = dm.get_prefix_lengths(df_train_bucket)[0] 
              #construct a df with case ids and variants they represent   
              file_name_prefix = '%s_%s_%s' %(ds, method_name, prfx_len)
              train_bucket = get_variants_df(variants_folder, \
                          'variants_and_weights_trainingLog_%s.csv' %file_name_prefix, df_train_bucket, dm, max_prefix_length_final)         
              test_bucket = get_variants_df(variants_folder, \
                          'variants_and_weights_testingLog_%s.csv' %file_name_prefix, df_test_bucket, dm, max_prefix_length_final)
