#Import Libraries
import pickle
import time
import os
import re
import sys
import operator
import string
import random
import pandas as pd
import numpy as np
import logging

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from helpers.Encoders import get_encoder
from helpers.Bucketers import get_bucketer
from helpers.DatasetManager import DatasetManager
from helpers.utils import  get_logger, save_pkl, discretize
from helpers.utils import get_samples, retrieve_artefact
from helpers.rules_utils import get_rules, clean_rules, decode_rule
from helpers.rules_utils import  match_vals, get_coverage

import lore
from anchor import utils
import anchor
from anchor import anchor_tabular



rules_dir = 'rules'
outcomes = 'artefacts'
variants_folder =os.path.join('Variants')
discretized_logs = os.path.join(outcomes, 'discretized_encoded_training')
params_dir = os.path.join(artefacts_dir,'cv_results')

cls_methods = ['logit',  'xgboost', 'gbm']
datasets = ['sepsis2', 'production', 'traffic_fines', 'BPIC2017_O_Accepted']
method_name = 'prefix_index'
bucket_method ='prefix'
cls_encoding = 'index'
bucket_encoding = "agg"
encoding_dict = { "agg": ["static", "agg"], "index": ["static", "index"]}
methods = encoding_dict[cls_encoding]
train_ratio = 0.8
random_state = 42 
gap = 1
n_iter = 5

size = {'sepsis2_single_agg' :500, 'production_single_agg':80, 'traffic_fines_single_agg':500, 'BPIC2017_O_Accepted_single_agg':500,
        'sepsis2_prefix_index':50, 'production_prefix_index':50, 'traffic_fines_prefix_index':200, 'BPIC2017_O_Accepted_prefix_index':200}


encoded_datasets_dir = 'encoded_datasets_%s' %(method_name)


for dataset_name in datasets:
    for cls_method in cls_methods:
        # for method_name in method_names:
        print('%s_%s_%s' % (dataset_name, cls_method, method_name))
        logger_name = os.path.join(outcomes, "executionLog_%s_%s.log" % (dataset_name, method_name))
        logger = get_logger(logger_name)

        # try:
        if method_name == "single_agg":
            gap = 1
        else:
            gap = 5

        params_file = os.path.join(params_dir,
                                   'optimal_params_%s_%s_%s.pickle' % (cls_method, dataset_name, method_name))
        with open(params_file, 'rb') as fin:
            args = pickle.load(fin)

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
                df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final,
                                                            gap=5)
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
                print('%s_%s' % (bucket_method, bucket))

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
                    pos_class_ratio = \
                    df_train_bucket.groupby(dm.case_id_col, as_index=False).first()[dm.label_col].value_counts()[
                        dm.pos_label] / len(df[dm.case_id_col].unique())
                    neg_class_ratio = \
                    df_train_bucket.groupby(dm.case_id_col, as_index=False).first()[dm.label_col].value_counts()[
                        dm.neg_label] / len(df[dm.case_id_col].unique())
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
                                                               learning_rate=args['learning_rate'],
                                                               max_depth=args['max_depth'],
                                                               subsample=args['subsample'],
                                                               colsample_bytree=args['colsample_bytree'],
                                                               min_child_weight=args['min_child_weight'],
                                                               seed=random_state)
                        elif cls_method == 'logit':
                            cls_experiment = LogisticRegression(C=2 ** args['C'], random_state=random_state)
                        elif cls_method == 'gbm':
                            cls_experiment = GradientBoostingClassifier(n_estimators=500,
                                                             learning_rate=args['learning_rate'],
                                                             subsample=args['subsample'],
                                                             min_samples_split=args['min_samples_split'],
                                                             min_samples_leaf=args['min_samples_leaf'],
                                                             max_features=args['max_features'],
                                                             random_state=random_state)

                        if cls_method  == "logit":
                            pipeline_final = Pipeline(
                                [('encoder', featureCombinerExperiment), ('scaler', StandardScaler()),
                                 ('cls', cls_experiment)])
                        else:
                            pipeline_final = Pipeline([('encoder', featureCombinerExperiment), ('cls', cls_experiment)])

                        pipeline_final.fit(df_train_bucket, train_y_experiment)
                        start = time.time()
                        preds = pipeline_final.predict(df_test_bucket)
                        end_pred = time.time() - start

                        ffeatures = pipeline_final.named_steps['encoder'].get_feature_names()
                        train_file_name = '%s_%s_%s_%s_%s_%s' % (
                        dataset_name, cls_method, method_name, df_train_bucket.shape[0], prfx_len, len(ffeatures))
                        test_file_name = '%s_%s_%s_%s_%s_%s' % (
                            dataset_name, cls_method, method_name, df_test_bucket.shape[0], prfx_len, len(ffeatures))
                        logger.info('got all the predictions for the test set: {0} in time {1}'.format(test_file_name,
                                                                                                       end_pred))

                        train_case_ids = pd.Series(df_train_bucket.groupby(dm.case_id_col).first().index).values
                        test_case_ids = pd.Series(df_test_bucket.groupby(dm.case_id_col).first().index).values


                        df_results = pd.DataFrame(
                            {"actual": test_y_real, "predicted": preds, "case_ids": test_case_ids,
                             'idx': [x for x in range(len(test_case_ids))]})

                        true = df_results[df_results["actual"] == df_results["predicted"]]
                        truly_predicted_indices = list(zip(true["case_ids"], true["predicted"], true['idx']))

                        logger.info(
                            'number of the truly predicted instances is {0}'.format(len(truly_predicted_indices)))
                        save_pkl(outputs, 'predictions_%s' % cls_method, train_file_name, preds)
                        save_pkl(outputs, 'truly_predicted_%s' % cls_method, train_file_name, truly_predicted_indices)

                        cat_cols = [*dm.static_cat_cols, *dm.dynamic_cat_cols]
                        categorical_feats = set([x for x in training_set_df.columns for y in cat_cols if y in x])
                        continuous_feats = set(training_set_df.columns) - set(categorical_feats)


                        discretized_train_df = discretize(discretized_logs, train_file_name, encoded_training, categorical_feats, ffeatures, train_case_ids, lbls, 'Training')
                        discretized_test_df = discretize(discretized_logs, test_file_name, encoded_testing, categorical_feats, ffeatures, test_case_ids, lbls, 'Testing')

                        variants_df = retrieve_artefact(variants_folder, '.csv',
                                                        'variants_and_weights_testingLog_%s_%s_%s' % (
                                                        dataset_name, method_name, prfx_len))

                        selected_cases_testing, selected_cases_data, selected_test_y = get_samples(execlog, variants_df,
                                                                                                   dm, df_test_bucket,
                                                                                                   truly_predicted_indices,
                                                                                                   size['%s_%s'] %(dataset_name, method_name))

                        df = pd.read_csv(os.path.join(encoded_datasets_dir, 'encoded_wholeDF_%s.csv' % train_file_name),
                                         sep=';')

                        columns = ffeatures
                        columns = ['label'] + columns
                        df = df[columns]
                        # possible_outcomes = list(df[class_name].unique())
                        possible_outcomes = [dm.pos_label, dm.neg_label]
                        type_features, features_type = recognize_features_type(df, 'label')

                        cat_cols = [col for col in columns if df[col].nunique() < 8]
                        num_cols = list(set(columns) - set(cat_cols))
                        discrete, continuous = set_discrete_continuous(columns, type_features, 'label',
                                                                       discrete=cat_cols, continuous=num_cols)

                        columns_tmp = list(columns)
                        columns_tmp.remove('label')
                        idx_features = {i: col for i, col in enumerate(columns_tmp)}

                        X = df[df.columns.difference(['label'])]

                        # Dataset Preparation for Scikit Alorithms
                        df_le, label_encoder = encoded_training, featureCombinerExperiment

                        y = train_y_experiment + test_y_real

                        dataset = {
                            'name': '%s_%s_%s_%s_%s' % (dataset_name, method_name, train_bkt_size, prfx_len, feat_num),
                            'df': df,
                            'columns': list(columns),
                            'class_name': 'label',
                            'possible_outcomes': possible_outcomes,
                            'type_features': type_features,
                            'features_type': features_type,
                            'discrete': discrete,
                            'continuous': continuous,
                            'idx_features': idx_features,
                            'label_encoder': label_encoder,
                            'X': X,
                            'y': y,
                        }

                        delimiter = ';'
                        y_idx = len(df.columns) - 1

                        data = os.path.join(encoded_datasets_dir, 'encoded_wholeDF_%s.csv' % train_file_name)

                        dataset = utils.load_csv_dataset(data, y_idx, delimiter, \
                                                         feature_names=ffeatures, categorical_features=categorical_feats);

                        explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, \
                                                                          dataset.train, dataset.categorical_names)


                        exp_dict_lore = {k: [] for k in
                                    ['row_idx',  'rule', 'anchor_pred', 'anchor_len']}

                        exp_dict_anchor = {k: [] for k in
                                         ['row_idx', 'explanation', 'prediction_explanation']}


                        operators = {'==': operator.eq, '>=': operator.ge, '<=': operator.le, '=>': operator.ge, \
                                     '=<': operator.le, '<': operator.lt, '>': operator.gt}
                        for row_name in selected_cases_testing:

                            case_loc = true.loc[true['case_ids'] ==row_name, 'idx']

                            logger.info('explaining:  %s' %row_name)

                            # get lore explanations
                            explanation, _ = lore.explain(case_loc, encoded_testing, dataset, cls_experiment,
                                                              ng_function=genetic_neighborhood,
                                                              discrete_use_probabilities=True,
                                                              continuous_function_estimation=False,
                                                              returns_infos=True,
                                                              path=path_data, sep=';', log=False)



                            exp_dict_lore['row_idx'].append(idx_record2explain)
                            exp_dict_lore['explanation'].append(explanation)
                            exp_dict_lore['prediction_explanation'].append(explanation[0][0])
                            clean_data = clean_rules(outputs, train_file_name + "_lore", explanation[0][1], operators,
                                                     'lore', logger)

                            #get anchor explanations
                            exp = explainer.explain_instance(df_test_bucket[case_loc], clf.predict, threshold=0.85)
                            anchor_pred = explainer.class_names[clf.predict(df_test_bucket[case_loc].reshape(1, -1))[0]]

                            exp_dict_anchor['row_idx'].append(row_name)
                            exp_dict_anchor['rule'].append(exp.names())
                            exp_dict_anchor['anchor_pred'].append(anchor_pred)
                            exp_dict_anchor['anchor_len'].append(len(exp.names()))

                            clean_data = clean_rules(outputs, train_file_name + "_anchor", exp.names(), operators,
                                                     'anchor', logger)

                        clean_lore_df(outcomes, exp_dict_lore, train_file_name, df_test_bucket, logger)
                        clean_anchor_df( outcomes, exp_dict_anchor, train_file_name, df_test_bucket, logger)





                        for rules_method in ['skope', 'rulefit', 'ripper']:
                          rules_folder = os.path.join(rules_dir, '%s' %rules_method)
                          #using RuleFit to extract rules:

                          if rules_method == 'rulefit':
                              logger.info('started rules generation using {0} in time {1}'.format(rules_method, time.time()))
                              start_rules = time.time()

                              train_rules_linear, trained_rulefit = get_rules(rules_folder, train_file_name, training_set_df, train_y_experiment,
                                                                              ffeatures,'rulefit', 'training')
                              end_rules = time.time() - start_rules

                              logger.info('Got rules from {0} in time {1}'.format(rules_method, end_rules))

                              train_rules = train_rules_linear[train_rules_linear['type']=='rule']
                              train_rules.sort_values(by=['importance'], ascending=False)
                              train_rules_list = train_rules['rule'].tolist()

                              #clean rules and extract them from strings:
                              logger.info('started rules cleaning using {0} in time {1}'.format(rules_method, time.time()))
                              start_cleaning = time.time()

                              cleaned_rules_df = clean_rules(rules_folder, train_file_name + "_rulefit", train_rules_list, operators, 'rulefit', logger, True)

                              end_cleaning = time.time() - start_cleaning
                              logger.info('cleaned rules from {0} in time {1}'.format(rules_method, end_cleaning))
                              rulefit_preds = trained_rulefit.predict(encoded_testing)
                              rule_preds = [1 if label == dm.pos_label else 0 for label in rulefit_preds]




                          elif rules_method == 'ripper':
                                 params_dict = {'X_test':testing_set_df, 'y_test': test_y, 'class_feat':dm.label_col, 'pos_class':minority_numeric}
                                 logger.info('started rules generation using {0} in time {1}'.format(rules_method, time.time()))
                                 start_rules = time.time()
                                 train_rules, trained_ripper = get_rules(rules_folder, train_file_name, training_set_df, train_y_experiment,
                                                                                  ffeatures,'ripper', 'training', params_dict)
                                 end_rules = time.time() - start_rules
                                 logger.info('Got rules from {0} in time {1}'.format(rules_method, end_rules))
                                 rules_str = str(trained_ripper.ruleset_).replace(']]',']').replace('[[','[').replace(']','').replace('[','').replace('^', ' & ')
                                 rules_lst = list(rules_str.split(" V "))

                                 logger.info('started rules cleaning using {0} in time {1}'.format(rules_method, time.time()))
                                 start_cleaning = time.time()

                                 cleaned_rules_df = clean_rules(rules_folder, train_file_name + "_ripper", rules_lst, operators, 'ripper', logger, True)

                                 end_cleaning = time.time() - start_cleaning
                                 logger.info('cleaned rules from {0} in time {1}'.format(rules_method, end_cleaning))
                                 ripper_preds = trained_ripper.predict(testing_set_df)
                                 rule_preds = [1 if label == dm.pos_label else 0 for label in ripper_preds]


                          elif rules_method == 'skope':
                                logger.info('started rules generation using {0} in time {1}'.format(rules_method, time.time()))
                                start_rules = time.time()

                                train_rules, trained_skope = get_rules(rules_folder, train_file_name, training_set_df, train_y_experiment, ffeatures,'skope', 'training')

                                end_rules = time.time() - start_rules
                                logger.info('Got rules from {0} in time {1}'.format(rules_method, end_rules))
                                rule_preds = trained_skope.predict_top_rules(encoded_testing, len(trained_skope.rules_))


                                logger.info('started rules cleaning using {0} in time {1}'.format(rules_method, time.time()))
                                start_cleaning = time.time()

                                cleaned_rules_df = clean_rules(rules_folder, train_file_name + "_skope", train_rules['original_rule'].tolist(), operators, 'skope', logger, True)

                                end_cleaning = time.time() - start_cleaning
                                logger.info('cleaned rules from {0} in time {1}'.format(rules_method, end_cleaning))



                          print('*************************************************************************************************************')

                          results_df = cleaned_rules_df[['splitted_rule']]
                          results_df.columns = ['clean_rule']
                          real_labels = {x: [] for x in results_df['clean_rule']}
                          rulemodel_preds = {x: [] for x in results_df['clean_rule']}


                          time_rules_method = []
                          matching_prediction_start = time.time()
                          for enc_rule_tuple in results_df['clean_rule']:
                              rule_tuple = decode_rule(enc_rule_tuple)

                              pred_times = []
                              for sidx_tup in selected_cases_testing:
                                  test_instance = test_buckets_grouped.get_group(sidx_tup)
                                  test_lbl = dm.get_label_numeric(test_instance)
                                  encoded_test_ins = featureCombinerExperiment.fit_transform(test_instance)

                                  test_ins = pd.DataFrame(encoded_test_ins, columns=ffeatures)

                                  count = 0
                                  for tup in rule_tuple:
                                      feat, operator, value = tup[0], operators[tup[1]], tup[2]
                                      if operator(test_ins[feat].values[0], value):
                                          count += 1  # counting the number of feats (in the instance) whose values meet the rule conditions

                                  if count == len(rule_tuple):
                                      # print(sidx_tup[0])
                                      matched_test_instances[enc_rule_tuple].append(sidx_tup)

                                      real_labels[enc_rule_tuple].append(test_lbl)

                                      start = time.time()
                                      pred_rulemodel = pred_instance(rule_model, encoded_test_ins, rules_method,
                                                                     ffeatures)
                                      end_pred = time.time() - start
                                      pred_times.append(end_pred)
                                      rulemodel_preds[enc_rule_tuple].append(pred_rulemodel)


                              time_rules_method.extend(pred_times)
                              logger.info('avg prediction time with rule {0} on {1} matched instances is {2}'.format(
                                  enc_rule_tuple, len(matched_test_instances[enc_rule_tuple]), np.mean(pred_times)))

                          matching_prediction_end = time.time() - matching_prediction_start

                          matched_test_instances, coverage_vol = get_coverage(cleaned_rules_df['splitted_rule'],
                                                                              discretized_test_df, logger, 'global',
                                                                              test_case_ids)
                          for k in matched_test_instances.keys():
                              results_df['matched_instances'] = results_df['clean_rule'].apply(
                                  lambda x: matched_test_instances[x])


                          results_df['coverage_vol'] = results_df['matched_instances'].apply(lambda x: len(x))
                          results_df['rule_length'] = results_df['clean_rule'].apply(lambda x: len(decode_rule(x)))


                          results_df['y_test'] = results_df['clean_rule'].apply(lambda x: real_labels[x])
                          results_df['pred_rule'] = results_df['clean_rule'].apply(lambda x: rulemodel_preds[x])

                          for clf_method in ['logit', 'gbm', 'xgboost']:
                              preds_file = open(
                                  os.path.join(bb_preds, 'predictions_%s_%s.pkl' % (clf_method, train_file_name)), 'rb')
                              preds_bb = pickle.load(preds_file)

                              clf_preds_col = 'preds_%s' % clf_method
                              results_df[clf_preds_col] = preds_bb

                              rule_preds_col = 'preds_%s' % rules_method
                              results_df[rule_preds_col] = rule_preds



                              match_label_col = 'match_%sVs%s' % (rules_method, clf_preds_col)
                              results_df[match_label_col] = results_df[[clf_preds_col, rule_preds_col]].apply(match_vals,
                                                                                                        axis=1)

                              save_pkl(outcomes, 'Global_and_preds_selectedInstances_', '%s_%s' % (train_file_name, clf_method),
                                       results_df)






    
   
    