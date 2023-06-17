import os
import re
import operator
import six
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
from rulefit import RuleFit
import wittgenstein as lw
from utils import save_pkl

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



