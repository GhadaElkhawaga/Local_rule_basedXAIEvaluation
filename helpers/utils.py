#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from numpy import percentile
import pandas as pd
import numpy as np
from lime.discretize import QuartileDiscretizer
import pickle
import logging


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger




def save_pkl(folder, elias, file_name, data):
    with open(os.path.join(folder, ('%s_%s.pkl' %(elias, file_name))), 'wb') as f:
            pickle.dump(data, f)
            f.close()
    return


def discretize(folder, file_name, encoded_data, categorical_feats, ffeatures, case_ids, labels, flag):
    discretizer = QuartileDiscretizer(encoded_data, categorical_feats, ffeatures, labels=labels, random_state=22)
    discretized_data = discretizer.discretize(encoded_data)
    discretized_data_df = pd.DataFrame(discretized_data, columns=ffeatures)
    discretized_data_df.index = case_ids
    save_pkl(folder, "discretizedDF" , '%s_%s' %(flag, file_name), discretized_data_df)
    #discretized_data_df.to_csv(os.path.join(discretized_logs, 'discretizedDF_%s_%s.csv' %(flag, file_name)), sep=';', index=False)
    return discretized_data_df



def get_samples(execlog, variants_df, dm, original_data, truly_predicted_indices, size):
    """
    a function for sampling an event log based on variants percentages
    parameters:
    - execlog: Execution log, used to write output messages during execution
    - variants_df: a dataframe containing case_ids along with their relevant variants and percentages, and other relevant information
    - dm: object from the data manager class
    - original_data : used to get instances selected to be in the sample event log
    - truly_predicted_indices : tuples of case_ids and indices of truly predicted instances
    outputs:
    - selected_instances: case_ids of instances selected to be in the sample event log
    - sampled_df_bucket: instances of the selected samples obtained from the original event log
    - sampled_y: values of the target class from the original event log
    - selected_samples_indices: indices of the selected samples
    """

    args = {'random_state' : 1, 'weights' : 'Percentage', 'n': size}
    sampled_set = variants_df.sample(**args)
    selected_instances_all = sampled_set[dm.case_id_col].tolist()
    true_cases = [x[0] for x in truly_predicted_indices]
    selected_instances = list(set(selected_instances_all) - set(true_cases))

    with open(execlog, 'a') as fout:
        fout.write('\n')
        fout.write('Selected instances: %s\n' %(selected_instances))
    selected_df_bucket = original_data[original_data[dm.case_id_col].isin(selected_instances)]
    selected_y = dm.get_label_numeric(selected_df_bucket)
    return selected_instances, selected_df_bucket,selected_y


#a function to retrieve artefacts
def retrieve_artefact(folder,file_end,*argv):
  retrieved_file = retrieve_file(folder,file_end,argv)
  if '.pickle' in file_end:
    with open(retrieved_file, 'rb') as fin:
        retrieved_artefact = pickle.load(fin)
  else:
    retrieved_artefact = pd.read_csv(retrieved_file,sep=';',encoding='ISO-8859-1')
  return retrieved_artefact

