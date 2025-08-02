import numpy as np
import matplotlib.pyplot as plt
import os
import math
from collections import defaultdict
from adjustText import adjust_text
from scipy.stats import linregress
import sys
import re

###########
# PARSING
###########

def parse_file_name(file_name, collapse_prompts=False):
    # filename has format d=<dataset>_m=<model>_q=<startq>-<endq>_c=<conf_type>_p=<prompt phrasing>.txt, possibly with _few_shot_<few shot num> at the end. We ignore the last part because we'll just rename the output files accordingly
    if 'few_shot' in file_name:
        few_shot_idx = file_name.find('_few_shot')
        file_name = file_name[:few_shot_idx] + '.txt'
    match = re.match(r'd=(.+?)_m=(.+?)_q=(\d+)-(\d+)_c=(.+?)_p=(.+?).txt', file_name)
    if not match:
        print(f"Error: Invalid file name format: {file_name}")
        sys.exit(1)
    dataset, model, start_q, end_q, conf_type, prompt_phrasing = match.groups()
    group = f"{conf_type},{prompt_phrasing}" if not collapse_prompts else f"{conf_type}"
    return dataset, model, group

def parse_group_name(group):
    parts = group.split(',')
    conf_type = parts[0] if len(parts) > 0 else None
    prompt_type = parts[1] if len(parts) > 1 else None
    confidence_measure = parts[2] if len(parts) > 2 else None
    renorm_status = parts[3] if len(parts) > 3 else None
    return conf_type, prompt_type, confidence_measure, renorm_status

def parse_data(file_path):
    labels = []
    conf_levels = []
    total_qs = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # skip header line
                if parts[0] not in ('Correct', 'Wrong'):
                    continue
                labels.append(1 if parts[0] == "Correct" else 0)
                confs_for_question = [float(x) for x in parts[1].split(',')]
                conf_levels.append(confs_for_question)
                total_qs += 1
    except IOError:
        print(f"Error opening file: {file_path}")
        sys.exit(1)
    return labels, conf_levels, total_qs

def groups_differ_by_one_component(group1, group2):
    """Check if two groups differ by exactly one component"""
    c1, p1, m1, r1 = parse_group_name(group1)
    c2, p2, m2, r2 = parse_group_name(group2)
    differences = int(c1 != c2) + int(p1 != p2) + int(m1 != m2) + int(r1 != r2 and r1 is not None and r2 is not None) # None counts as a match                                    
    return differences == 1


###################
# CONVERTING TO/FROM HUMAN READABLE
###################

def expand_model_name(name):
    main_name = name.split('-base')[0] # ignore whether it's a base or chat model; it will be clear from the caption
    return ('Mistral 7B' if main_name == 'Mistral' else
            'Mixtral 8x7B' if main_name == 'Mixtral' else
            'SOLAR 10.7B' if main_name == 'Solar' else
            'Llama 2 13B' if main_name == 'Llama2-13b' else
            'Llama 2 7B' if main_name == 'Llama2-7b' else
            'Llama 2 70B' if main_name == 'Llama2-70b' else
            'Llama 3.0 8B' if main_name == 'Llama3-8b' else
            'Llama 3.0 70B' if main_name == 'Llama3-70b' else
            'Llama 3.1 8B' if main_name == 'Llama3.1-8b' else
            'Llama 3.1 70B' if main_name == 'Llama3.1-70b' else
            'Yi 6B' if main_name == 'Yi-6b' else
            'Yi 34B' if main_name == 'Yi-34b' else
            'GPT-3.5 Turbo' if main_name == 'gpt-3.5-turbo' else
            'GPT-4o' if main_name == 'gpt-4o' else
            'Falcon 7B' if main_name == 'Falcon-7b' else
            'Falcon 40B' if main_name == 'Falcon-40b' else main_name)

def expand_label(label):
        return ('Confidence threshold' if label == 'conf' else
                'Score (balanced)' if label == 'score' else
                'Score (conservative)' if label == 'harsh-score' else
                'Model size (billions of parameters)' if label == 'size' else
                'AUROC' if label == 'auc' else
                'Fraction correct' if label == 'frac-correct' else
                'MSP' if label == 'msp' else
                'Calibration error (x100)' if label == 'calib' else
                'Calibration error (x100, Platt scaling)' if label == 'calib-platt' else
                'Number of training examples' if label == 'num_train' else
                'Q&A accuracy' if label == 'acc' else label)

def group_label_human_readable(group):
    conf_type, prompt_type, confidence_measure, renorm_status = parse_group_name(group)
    
    if conf_type == 'prob':
        conf_label = 'MSP'
    elif conf_type == 'logit':
        conf_label = 'Max Logit'
    else:
        conf_label = conf_type or ''
    
    if confidence_measure == 'max':
        measure_label = '' 
    elif confidence_measure == 'margin':
        measure_label = ' Margin'
    elif confidence_measure == 'entropy':
        measure_label = ' Entropy'
    else:
        measure_label = ''
    
    if renorm_status == 'renorm':
        renorm_label = ''
    elif renorm_status == 'no_renorm':
        renorm_label = ' (unnormalized)'
    else:
        renorm_label = ''
    
    if prompt_type == 'first_prompt':
        prompt_label = ' first phrasing'
    elif prompt_type == 'second_prompt':
        prompt_label = ' second phrasing'
    else:
        prompt_label = ''
    
    return conf_label + measure_label + renorm_label + prompt_label

def sanitize_group_name_for_path(group_name):
    # Replace commas with underscores. If there's a double comma (happens with collapse_prompts=True), replace with single underscore
    return group_name.replace(',,', '_').replace(',', '_')

def get_file_suffix(dataset, group_name, num_train=None):
    sanitized_group_name = sanitize_group_name_for_path(group_name)
    file_suffix = f'_{dataset}_{sanitized_group_name}'
    if num_train is not None:
        file_suffix += f'_k{num_train}'
    if '_renorm' in file_suffix:
        file_suffix = file_suffix.replace('_renorm', '') # renorm is default
    while '__' in file_suffix:
        file_suffix = file_suffix.replace('__', '_')
    if file_suffix.endswith('_'):
        file_suffix = file_suffix[:-1]
    return file_suffix

# Each model name is of the form "<model_series> <size>B" or "<model_series> <size>B base". Model series could have spaces also (e.g., Llama 2). Mixtral is a slight exception, as are OpenAI gpt models, which have no size term
def model_series(name):
    # remove base, which is irrelevant here
    name = name.replace('-base', '')
    expanded_name = expand_model_name(name)
    return expanded_name[:expanded_name.rfind(' ')]

def model_size(name):
    if 'Mixtral' in name:
        return 46.7
    elif 'gpt' in name.lower():
        return -1
    else:
        full_name = expand_model_name(name)
        end_of_size_term = full_name.rfind('B')
        main_name = full_name[:end_of_size_term]
        size_term = main_name.split(' ')[-1]
        return float(size_term)

def format_dataset_name(dataset):
    if dataset.endswith('_'):
        dataset = dataset[:-1]  # Remove trailing underscore if present
    return ('ARC-Challenge' if dataset == 'arc' else
            'HellaSwag' if dataset == 'hellaswag' else
            'MMLU' if dataset == 'mmlu' else
            'TruthfulQA' if dataset == 'truthfulqa' else
            'WinoGrande' if dataset == 'winogrande' else dataset)

def dataset_tag(dataset_list):
    if len(dataset_list) == 1: # specific dataset, tag it
        return dataset_list[0] + '_'
    else:
        return ''

###############
# DATA PROCESSING
###############

def extract_confidence_measure(distribution, measure_type, renormalize=True):
    """Extract confidence measure from a probability/logit distribution"""
    if len(distribution) == 0:
        raise ValueError("Distribution cannot be empty")
    
    if renormalize:
        # Only renormalize for max/margin, not for entropy (handle separately)
        dist_sum = sum(distribution)
        if dist_sum <= 1e-15:
            raise ValueError("Cannot renormalize distribution with sum <= 0")
        distribution = [x/dist_sum for x in distribution]
    
    if measure_type == 'max':
        return max(distribution)
    elif measure_type == 'margin':
        if len(distribution) < 2:
            raise ValueError("Margin requires at least two elements in the distribution")
        sorted_dist = sorted(distribution, reverse=True)
        return sorted_dist[0] - sorted_dist[1]
    elif measure_type == 'entropy':
        # switched sign because we want higher vals to mean more confidence
        return 1 + sum(p * math.log(p) for p in distribution if p > 1e-15)  # Numerical stability
    else:
        raise ValueError(f"Unknown measure_type: {measure_type}")

def generate_all_confidence_variants(labels, conf_levels_distributions, total_qs, conf_type, prompt_type):
    variants = {}
    
    if conf_type == 'logit':
        measures = ['max', 'margin']
        for measure in measures:
            extracted = [extract_confidence_measure(dist, measure, False) for dist in conf_levels_distributions]
            group_name = f"{conf_type},{prompt_type},{measure}"
            variants[group_name] = (labels, extracted, total_qs)
    
    elif conf_type == 'prob':
        measures = ['max', 'margin']
        for measure in measures:
            for renorm in [True]: # always renormalize distribution for now
                extracted = [extract_confidence_measure(dist, measure, renorm) for dist in conf_levels_distributions]
                renorm_suffix = 'renorm' if renorm else 'no_renorm'
                group_name = f"{conf_type},{prompt_type},{measure},{renorm_suffix}"
                variants[group_name] = (labels, extracted, total_qs)
        
        # Entropy only for renormalized
        extracted = [extract_confidence_measure(dist, 'entropy', True) for dist in conf_levels_distributions]
        group_name = f"{conf_type},{prompt_type},entropy,renorm"
        variants[group_name] = (labels, extracted, total_qs)
    
    return variants

def collapse_data_to_model(data, duplicate_data=True):
    # Input: dict of the form data[dataset][model] = (labels, conf_levels, total_qs)
    # Output: dict of the form data[model] = (labels, conf_levels, total_qs).
    # If duplicate_data, we duplicate questions from smaller datasets so that they have the same total number as the largest datasets
    collapsed = defaultdict(lambda: ([], [], 0))
    target_num_questions = max([data[dataset][model][2] for dataset in data for model in data[dataset]])
    for dataset in data:
        for model in data[dataset]:
            (labels, conf_levels, total_qs) = data[dataset][model]
            deterministic_dup_factor = target_num_questions // len(labels)
            extra_dups_needed = target_num_questions % len(labels)
            new_labels = np.tile(labels, deterministic_dup_factor)
            new_conf_levels = np.tile(conf_levels, deterministic_dup_factor)
            # randomly choose an additional extra_dups_needed questions so we hit target_num_questions
            np.random.seed(2549900867)
            random_indices = np.random.choice(len(labels), extra_dups_needed, replace=False)
            new_labels = np.concatenate([new_labels, labels[random_indices]])
            new_conf_levels = np.concatenate([new_conf_levels, conf_levels[random_indices]])
            total_qs = len(new_labels)
            assert(total_qs == target_num_questions), f"Expected {target_num_questions} questions, got {total_qs}"
            (old_labels, old_conf_levels, old_total_qs) = collapsed[model]
            collapsed[model] = (np.concatenate([old_labels, new_labels]), np.concatenate([old_conf_levels, new_conf_levels]), old_total_qs + total_qs)
    return collapsed

def merge_group_metrics(group_metrics):
    # Merge to a single "group" based on the means across groups
    all_auc_calib_acc_data = defaultdict(lambda: ([], [], []))
    all_score_data = defaultdict(lambda: (defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)))
    for group in group_metrics:
        (score_data, aucs, calib_errors, accs, model_names) = group_metrics[group]
        for i, model_name in enumerate(model_names):
            # Collect the auc and acc for this model in each group into a list
            all_auc_calib_acc_data[model_name][0].append(aucs[i])
            all_auc_calib_acc_data[model_name][1].append(calib_errors[i])
            all_auc_calib_acc_data[model_name][2].append(accs[i])

        if score_data is not None:
            for wrong_penalty in score_data:
                # Same idea here, except these are each dicts with model name as the key
                (thresholds, our_scores, base_scores, pcts_abstained) = score_data[wrong_penalty]
                for model in thresholds:
                    all_score_data[wrong_penalty][0][model].append(thresholds[model])
                    all_score_data[wrong_penalty][1][model].append(our_scores[model])
                    all_score_data[wrong_penalty][2][model].append(base_scores[model])
                    all_score_data[wrong_penalty][3][model].append(pcts_abstained[model])

    avg_aucs, avg_calib_errors, avg_accs, model_names = [], [], [], []
    for model_name, (aucs, calib_errors, accs) in all_auc_calib_acc_data.items():
        avg_aucs.append(np.mean(aucs))
        avg_calib_errors.append(np.mean(calib_errors))
        avg_accs.append(np.mean(accs))
        model_names.append(model_name)

    new_score_data = defaultdict(lambda: (dict(), dict(), dict(), dict()))
    for wrong_penalty, (thresholds, our_scores, base_scores, pcts_abstained) in all_score_data.items():
        for model in thresholds:
            thresh_list, our_scores_list, base_scores_list, pct_abstained_list = thresholds[model], our_scores[model], base_scores[model], pcts_abstained[model]
            new_thresh, new_our_score, new_base_score, new_pct_abstained = np.mean(thresh_list), np.mean(our_scores_list), np.mean(base_scores_list), np.mean(pct_abstained_list)
            new_score_data[wrong_penalty][0][model] = new_thresh
            new_score_data[wrong_penalty][1][model] = new_our_score
            new_score_data[wrong_penalty][2][model] = new_base_score
            new_score_data[wrong_penalty][3][model] = new_pct_abstained
    return new_score_data, avg_aucs, avg_calib_errors, avg_accs, model_names

def make_model_dict(score_data, aucs, calib_errors, accs, model_names):
    # Change the dict structure so that the model is the key
    model_results = dict()
    for i, model_name in enumerate(model_names):
        model_score_data = dict()
        for wrong_penalty in score_data:
            thresholds, our_scores, base_scores, pcts_abstained = score_data[wrong_penalty]
            thresh, our_score, base_score, pct_abstained = thresholds[model_name], our_scores[model_name], base_scores[model_name], pcts_abstained[model_name]
            model_score_data[wrong_penalty] = (thresh, our_score, base_score, pct_abstained)
        model_results[model_name] = (aucs[i], calib_errors[i], accs[i], model_score_data)
    return model_results

                        
def gather_all_data(output_dir, datasets_to_analyze, collapse_prompts, file_paths):
    # We want all_data[group][dataset][model] = (labels, conf_levels, total_qs)
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], 0))))
    
    for file_path in file_paths:
        base_group_name = parse_file_name(os.path.basename(file_path), collapse_prompts)
        if len(base_group_name) == 3:
            dataset, model, group = base_group_name
            if dataset in datasets_to_analyze:
                # Parse the original group format to get conf_type and prompt_type
                if ',' in group:
                    conf_type, prompt_type = group.split(',')
                else:
                    conf_type = group
                    prompt_type = ''  # Default for collapsed prompts
                labels, conf_levels_dists, total_qs = parse_data(file_path)
                variants = generate_all_confidence_variants(labels, conf_levels_dists, total_qs, conf_type, prompt_type)
                # Add each variant to all_data
                for variant_group_name, (variant_labels, variant_conf_levels, variant_total_qs) in variants.items():
                    old_labels, old_conf_levels, old_total_qs = all_data[variant_group_name][dataset][model]
                    all_data[variant_group_name][dataset][model] = (np.concatenate([old_labels, variant_labels]), np.concatenate([old_conf_levels, variant_conf_levels]), old_total_qs + variant_total_qs)
    return all_data

##############
# MISC
##############

def sort_models(models):
    # Sort by model series, then by model size. Also put OpenAI gpt models at the end
    return sorted(models, key=lambda x: ('gpt' in x, model_series(x), model_size(x)))

def make_pct(x):
    return 100*x

def str_to_bool(s):
    if s.lower() in ('true', 'yes', 'y', '1'):
        return True
    elif s.lower() in ('false', 'no', 'n', '0'):
        return False
    else:
        raise Exception("Unrecognized boolean string")
