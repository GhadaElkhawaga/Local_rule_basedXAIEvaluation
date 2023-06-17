import os
import re
import operator
import six
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
from rulefit import RuleFit
import wittgenstein as lw
from utils import save_pkl


def decode_rule(rule_tuple):
    converted_rule = re.findall(r'\(.*?\)', rule_tuple)
    try:
        converted_rule = [eval(el) for el in converted_rule]
        converted_rule = [(elm[0], elm[1], eval(elm[2])) for elm in converted_rule]
    except:
        converted_rule = [tuple(el) for el in converted_rule]

    converted_rule = [tuple(xi for xi in x if xi is not None) for x in converted_rule]
    return converted_rule



def match_vals(row):
    if row[0] == row [1]:
        return 1
    else:
        return 0


def get_rules(folder, file_name, X, y, ffeatures, mode, flag, args=None):
    if mode == 'rulefit':
        rules_model = RuleFit(rfmode='classify',
                              tree_generator=GradientBoostingClassifier(n_estimators=500, random_state=22) )
        rules_model.fit(X.values, y, feature_names=ffeatures)
        rules = rules_model.get_rules()
        rules = rules[(rules.support > 0.8 ) &(rules.coe f! =0)].sort_values("support", ascending=False)
        saved_items = {'elias' :['Rulefitrules_%s' %file_name, 'RulefitModel_%s' % file_name],
                       'content': [rules, rules_model]}

    elif mode == 'ripper':
        rules_model = lw.RIPPER()

        rules_model.fit(X, y, feature_names=ffeatures, pos_class=args['pos_class'])
        rules = str(rules_model.ruleset_)

        accuracy = rules_model.score(args['X_test'], args['y_test'])
        precision = rules_model.score(args['X_test'], args['y_test'], precision_score)
        recall = rules_model.score(args['X_test'], args['y_test'], recall_score)
        performance = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
        rules_model.out_model()


        saved_items = {
            'elias': ['Ripperrules_%s' % file_name, 'RipperModel_%s' % file_name, 'Ripperperformance_%s' % file_name],
            'content': [rules, rules_model, performance]}

    elif mode == 'skope':
        # Train a skope-rules-boosting classifier
        rules_model = SkopeRules(feature_names=ffeatures, random_state=42, n_estimators=50,
                                 recall_min=0.05, precision_min=0.9,
                                 max_samples=0.7,
                                 max_depth_duplication=4, max_depth=5)
        rules_model.fit(X, y)
        # rules_model.rules_ outputs: dict of tuples (rule, precision, recall, nb).
        rules = pd.DataFrame(rules_model.rules_, columns=['original_rule', 'performance'])

        rules[['precision', 'recall', 'nb']] = pd.DataFrame(rules['performance'].tolist(), index=rules.index)

        saved_items = {'elias': ['Skoperules_%s_original_withperformance' % file_name, 'SkopeModel_%s' % file_name],
                       'content': [rules, rules_model]}

    for i in range(len(saved_items.keys())):
        save_pkl(folder, saved_items['elias'][i], flag, saved_items['content'][i])
    return rules, rules_model


def preprocess_tuple(rule_part, flag):
    tmp_rule_part = list(rule_part)
    if flag == 'feat_clean':
        if len(tmp_rule_part) == 3:  # in case of a single condition subrule, ex: age > 50
            idx = 0
        elif len(tmp_rule_part) == 5:  # in case of a range-based subrule, ex: 20 < age <= 65
            idx = 2
        tmp_rule_part[idx] = tmp_rule_part[idx][1:].replace("\'",
                                                            "")  # to clean features from byte characters and quotations
    elif flag == 'operator_clean':
        idx = 1
        tmp_rule_part[idx] = '=='  # to transform the '=' operator into '=='
    rule_part = tuple(tmp_rule_part)
    return rule_part


def split_rules(j, operators, logger):

    # split the rules into
    for op in operators:
        if re.search(op, j):
            res_rule_part = tuple(map(str.strip, j.partition(op)))
            tmp_rule_part = list(res_rule_part)
            # to catch and split again if the rule follows this format: 20 < age <= 65
            for op in operators:
                if re.search(op, tmp_rule_part[2]):
                    sub_rule_part = tuple(map(str.strip, tmp_rule_part[2].partition(op)))
                    tmp_rule_part.pop(2)
                    tmp_rule_part.extend(list(sub_rule_part))
                    # tmp_rule_part = [x.replace('[','').replace(']','') for x in tmp_rule_part]
                    rule_part = tuple(tmp_rule_part)
                    break
                else:
                    rule_part = tuple(tmp_rule_part)

            return rule_part
    if re.search('=', j):
        rule_tup = tuple(map(str.strip, j.partition('=')))
        rule_part = preprocess_tuple(rule_tup, 'operator_clean')

    else:
        rule_part = j


    return rule_part


def compare_rules(train_rules, test_rules):
    common_rules = list(set(train_rules['rule'].tolist()) & set(test_rules['rule'].tolist()))
    common_df = test_rules[test_rules['rule'].isin(common_rules)].sort_values("importance", ascending=False)
    return common_df


def select_rules(rules_df, execlog):
    rules_df['#feats'] = list(map(lambda x: x.count('&'),rules_df['rule']))
    if rules_df.shape[0] <= 10:
        with open(execlog, 'a') as fout:
            fout.write('\n')
            fout.write('Rules are less than 10, then we selected them all. \n')
            fout.write('Selected rules: %s\n' %(rules_df.loc[rules_df['type']=='rule','rule'].tolist()))
        return rules_df.loc[rules_df['type']=='rule','rule'].tolist()
    else:
        #to select rules including conditions on multiple features
        selected_rules = rules_df[rules_df['#feats']>1]['rule'].values
        if len(selected_rules) == 0:
            with open(execlog, 'a') as fout:
                fout.write('All common rules consist of one term')
            return rules_df['rule'].tolist()
        else:
            with open(execlog, 'a') as fout:
                fout.write('\n')
                fout.write('Selected rules: %s\n' %(selected_rules))
            return selected_rules


def match_rules_instances(selected_correctpred_cases,encoded_selected_cases_data,rules_df):
    rules_met = []
    for instance in selected_correctpred_cases.itertuples():
        process_instance_data = encoded_selected_cases_data[encoded_selected_cases_data[dm.case_id_col] == getattr(instance,dm.case_id_col)].head(1)
        #process_instance_data = selected_cases_data[selected_cases_data[dm.case_id_col]==instance].head(1)
        #encoded_process_instance_data = featureCombinerExperiment.transform(process_instance_data)
        match = []
        for row in rules_df.itertuples():
            feats_tuples = getattr(row,'cleaned_rules')
            count = 0
            for tup in feats_tuples:
                feat, operator, value = tup[0], operators[tup[1]], tup[2]
                if operator(encoded_process_instance_data.loc[:,feat], value):
                    count += 1 #counting the number of feats (in the instance) whose values meet the rule conditions
            if count == len(feats_tuples):
                match.append(getattr(row,'cleaned_rules'))
            else:
                match.append('not all conditions matched')
        rules_met.append(match)
    selected_correctpred_cases['RulesMet'] = rules_met #recording all rules met by a certain process instance
    return selected_correctpred_cases



def parse_str(rule):

    if rule != 'no matching rule':
        rule = rule.split("None")[0]
        rule_lst = [x for x in eval(rule)]
        feats = [i[0] for i in rule_lst]
    else:
        feats = 'no matching rule'
    return feats


def get_rules_df(rules_dict, rules_len, mode):
    rules_df = pd.DataFrame.from_dict(rules_dict, orient='index')
    rules_df['length'] = list(rules_len.values())
    rules_df = rules_df.rename_axis('originalRule').reset_index()
    rules_df = rules_df.astype({col: str for col in rules_df.columns.tolist() if col not in ['length']})
    if mode in ['rulefit', 'ripper', 'skope']:
        combined_cols = rules_df.columns.tolist()
        combined_cols.remove('originalRule')
        combined_cols.remove('length')
        rules_df['splitted_rule'] = rules_df[combined_cols].apply(lambda x: ",".join(x), axis=1)
        rules_df = rules_df[['originalRule', 'length', 'splitted_rule']]
        rules_df['%s_expRuleFeats' % mode] = rules_df['splitted_rule'].apply(parse_str)

    return rules_df


def get_subrules_tuple(sub_rules_list, str_operators, logger):
    subrules_tup_lst = []
    for j in sub_rules_list:
        rule_tuple = split_rules(j, list(str_operators), logger)
        subrules_tup_lst.append(rule_tuple)
    return subrules_tup_lst


def clean_rules(folder, file_name, rules, operators, mode, logger, save_flag=False):
    if mode in ['rulefit', 'ripper', 'skope']:

        rules_dict, rules_len = {}, {}
        # checking rules with multiple conditions on multiple features
        for i in range(len(rules)):
            if mode == 'skope':
                rules[i] = rules[i].replace('and', '&')
            if re.search('&', rules[i]):
                sub_rules_list = rules[i].split('&')
                rules_len[rules[i]] = len(sub_rules_list)
                subrules_tup_lst = get_subrules_tuple(sub_rules_list, operators.keys(), logger)
                rules_dict[rules[i]] = subrules_tup_lst
            else:
                rules_len[rules[i]] = 1
                rules_dict[rules[i]] = [split_rules(rules[i], list(operators.keys()), logger)]
        cleaned_rules = get_rules_df(rules_dict, rules_len, mode)
        if save_flag:
            save_pkl(folder, 'PartitionedRules_train', file_name, cleaned_rules)
        # cleaned rules in a dataframe in case of rulefit
        artefacts = cleaned_rules

    elif mode == 'anchor':
        anchor_subrules_tup_lst = get_subrules_tuple(rules, operators.keys())
        original_rule = '[' + ','.join(rules) + ']'
        cleaned_rules = [preprocess_tuple(x, 'feat_clean') for x in anchor_subrules_tup_lst]
        length = len(cleaned_rules)
        # a list of cleaned_rules with their length in case of anchor
        artefacts = [cleaned_rules, length]

    elif mode == 'LORE':
        rules_dict, rules_len = {}, {}
        for i in range(len(rules)):
            rules_len[i] = len(rules[i])
            intermediate = {}
            for k, v in rules[i].items():
                cut = []
                for op in list(operators.keys()):
                    if re.search(op, v):
                        cut = [op, v.split(op)[1]]
                        break
                if not cut:
                    cut = ['==', v]
                intermediate[k] = (k, cut[0], cut[1])
            rules_dict[i] = list(intermediate.values())
        artefacts = [rules_dict, rules_len]

    return artefacts


def get_coverage(rules, test_data_df, logger, fname, flag, test_cases=None):

    if flag in ['lore', 'anchor']:
        matched_test_instances = {str(x): [] for x in rules}
    else:
        matched_test_instances = {x: [] for x in rules}
    coverage_vol = {}
    operators = {'==': operator.eq, '>=': operator.ge, '<=': operator.le, '=>': operator.ge, '=<': operator.le,
                 '<': operator.lt, '>': operator.gt}
    test_cases = list(test_data_df.index.values)

    for enc_rule_tuple in rules:
        if flag in ['lore', 'anchor']:
            rule_tuple = decode_rule(str(enc_rule_tuple))
        else:
            rule_tuple = decode_rule(enc_rule_tuple)

        if not rule_tuple:
            if flag in ['lore', 'anchor']:
                matched_test_instances[str(enc_rule_tuple)] = []
                coverage_vol[str(enc_rule_tuple)] = 0
            else:
                matched_test_instances[enc_rule_tuple] = []
                coverage_vol[enc_rule_tuple] = 0
            logger.info('rule {0} matches 0 instances'.format(rule_tuple))
            continue

        matched_ins_count = 0
        for test_idx in test_cases:

            encoded_test_ins = test_data_df.loc[test_idx]
            test_ins = encoded_test_ins.to_frame().T

            count = 0
            for tup in rule_tuple:

                if len(tup) == 3:
                    if isinstance(tup[0], str) and isinstance(tup[2], numbers.Number):
                        feat, operator, value = tup[0], operators[tup[1]], tup[2]
                    elif isinstance(tup[2], str) and isinstance(tup[0], numbers.Number):
                        value, operator, feat = tup[0], operators[tup[1]], tup[2]
                    if all([x in fname for x in ['traffic_fines_prefix_index_8736_6_901', 'gbm']]) or all(
                            [x in fname for x in ['traffic_fines_single_agg_362094_1_254', 'xgboost']]) or all(
                            [x in fname for x in ['traffic_fines_single_agg_362094_1_254', 'gbm']]) or all(
                            [x in fname for x in ['sepsis2_prefix_index_468_11_408', 'xgboost']]):

                        try:
                            feat, operator, value = tup[0], operators[tup[1]], eval(tup[2])
                        except:
                            feat, operator, value = tup[0], operators[tup[1]], tup[2]

                        if feat.startswith('gg'):
                            feat = feat.replace('gg', 'agg')
                        elif feat.startswith('tatic'):
                            feat = feat.replace('tatic', 'static')
                        elif feat.startswith('ndex'):
                            feat = feat.replace('ndex', 'index')

                    if operator(test_ins[feat].values[0], value):
                        count += 1  # counting the number of feats (in the instance) whose values meet the rule conditions

                elif len(tup) == 5:
                    val1, op1, feat, op2, val2 = eval(tup[0]), operators[tup[1]], tup[2], operators[tup[3]], eval(
                        tup[4])
                    if feat.startswith('gg'):
                        feat = feat.replace('gg', 'agg')
                    elif feat.startswith('tatic'):
                        feat = feat.replace('tatic', 'static')
                    elif feat.startswith('ndex'):
                        feat = feat.replace('ndex', 'index')

                    if op1(test_ins[feat].values[0], val1) and op2(test_ins[feat].values[0], val2):
                        count += 1

            if count == len(rule_tuple):
                matched_ins_count += 1
                if flag in ['lore', 'anchor']:
                    matched_test_instances[str(enc_rule_tuple)].append(test_idx)
                else:
                    matched_test_instances[enc_rule_tuple].append(test_idx)

        if flag in ['lore', 'anchor']:
            coverage_vol[str(enc_rule_tuple)] = matched_ins_count
        else:
            coverage_vol[enc_rule_tuple] = matched_ins_count

        logger.info('rule {0} matches {1} instances'.format(rule_tuple, matched_ins_count))

    return matched_test_instances, coverage_vol


def get_match_preds(in_dir, file_name, df, flag):
    for clf_method in ['xgboost', 'logit', 'gbm']:
        global_data = pickle.load(
            open(os.path.join(in_dir, 'Global_and_preds_selectedInstances_%s_%s.pkl' % (clf_method, file_name)), 'rb'))

        df[['actual', 'preds_%s' % clf_method]] = global_data.loc[
            global_data['case_ids'].isin(df['row_idx'].values.tolist()), ['actual', 'preds_%s' % clf_method]]
        df[['actual', 'preds_%s' % clf_method]] = df[['actual', 'preds_%s' % clf_method]].fillna('')

        for col in ['actual', 'preds_%s' % clf_method]:
            match_label_col = 'match_%sVs%s' % (flag, col)
            if 'production' in file_name:
                print('from inside get match preds')
                print(df[[col, '%s_pred' % flag]])
            df[match_label_col] = df[[col, '%s_pred' % flag]].apply(match_vals, axis=1)
            df.drop(col, axis=1, inplace=True)

    return df


def clean_lore_df(lore_dir, lore_dict, file_name, test_data, logger):
    logger.info('now working on %s in lore' % file_name)
    lore_df = pd.DataFrame(lore_dict, columns=['row_idx', 'explanation'])
    lore_df['lore_pred'] = lore_df['explanation'].apply(lambda x: x[0][0]['encoded_label'])
    lore_df['lore_pred'] = lore_df['lore_pred'].astype(int)
    lore_df['lore_expRule'] = lore_df['explanation'].apply(lambda x: x[0][1])
    lore_df['lore_expRuleFeats'] = lore_df['lore_expRule'].apply(lambda x: list(x.keys()))
    lore_df['lore_expRuleCleaned'] = lore_df['lore_expRule'].apply(lambda x: clean_rules(x, 'LORE'))
    lore_df['lore_LenExpRule'] = lore_df['explanation'].apply(lambda x: len(x[0][1]))

    fname = clean_name(file_name)
    lore_df = get_match_preds(lore_dir, fname, lore_df, 'lore')
    matched_test_instances, coverage_vol = get_coverage(lore_df['lore_expRuleCleaned'], test_data, logger, file_name,
                                                        'lore')

    lore_df['matched_instances'] = lore_df['lore_expRuleCleaned'].apply(lambda x: matched_test_instances[str(x)])
    lore_df['coverage_vol'] = lore_df['lore_expRuleCleaned'].apply(lambda x: coverage_vol[str(x)])

    coverage_len_corr = lore_df['coverage_vol'].corr(lore_df['lore_LenExpRule'])

    print(file_name)
    print(lore_df)
    print('********************************************************************************************')
    save_pkl(lore_dir, 'prefinal_loreExp', file_name, lore_df)

    return coverage_len_corr


def clean_anchor_df(anchor_dir, anchor_dict, file_name, test_data, logger):
    logger.info('now working on %s in anchor' % file_name)
    anchor_df = pd.DataFrame(anchor_dict, columns=['row_idx', 'rule', 'anchor_pred', 'anchor_len'])
    anchor_df['anchor_pred_cleaned'] = anchor_df['anchor_pred'].apply(lambda x: int(eval(x)))
    anchor_df.drop(['anchor_pred'], axis=1, inplace=True)
    anchor_df.rename(
        columns={'anchor_pred_cleaned': 'anchor_pred', 'rule': 'anchor_expRule', 'anchor_len': 'anchor_LenExpRule'},
        inplace=True)
    anchor_df['anchor_expRuleCleaned'] = anchor_df['anchor_expRule'].apply(lambda x: clean_rules(x, 'anchor'))
    anchor_df['anchor_expRuleFeats'] = anchor_df['anchor_expRuleCleaned'].apply(lambda x: [i[0] for i in x])

    fname = clean_name(file_name)
    anchor_df = get_match_preds(anchor_dir, fname, anchor_df, 'anchor')

    matched_test_instances, coverage_vol = get_coverage(anchor_df['anchor_expRuleCleaned'], test_data, logger,
                                                        file_name, 'anchor')
    anchor_df['matched_instances'] = anchor_df['anchor_expRuleCleaned'].apply(lambda x: matched_test_instances[str(x)])
    anchor_df['coverage_vol'] = anchor_df['anchor_expRuleCleaned'].apply(lambda x: coverage_vol[str(x)])

    coverage_len_corr = anchor_df['coverage_vol'].corr(anchor_df['anchor_LenExpRule'])

    print(file_name)
    print(anchor_df)
    print('********************************************************************************************')
    save_pkl(anchor_dir, 'prefinal_anchorExp', file_name, anchor_df)
    return coverage_len_corr
