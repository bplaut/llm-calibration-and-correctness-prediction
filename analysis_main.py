import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
import os
import math
from collections import defaultdict
import random
from utils import *
from plotting_functions import *

def plots_for_group(data, output_dir, collapse_prompts, group, num_train, platt):
    arbitrary_dataset = next(iter(data)) # Get an arbitrary dataset to use for model names
    model_names = sort_models(data[arbitrary_dataset].keys())
    score_data, aucs, calib_errors, accs = {}, [0] * len(model_names), [0] * len(model_names), [0] * len(model_names) # We won't always populate all of these (depending on collapse_prompts) but we need them for the return value
    datasets = list(data.keys())
    group_tag = sanitize_group_name_for_path(group)
    train_data, test_data = split_into_train_and_test(data, num_train)
    if platt:
        group_tag += '_platt'
        collapsed_train_data = collapse_data_to_model(train_data)
        platt_params = {model: fit_platt_scaling(labels, scores) for
                        model, (labels, scores, _) in collapsed_train_data.items()}
    # If collapse_prompts, we do Q&A with abstention and calibration curves.
    # Otherwise, we do AUROC and calibration error scatter plots
    if collapse_prompts:
        # Q&A with abstention
        wrong_penalties = [1, 2] # 1 for balanced, 2 for conservative
        score_data = {wrong_penalty: train_and_test_score_plots(test_data, train_data, output_dir, datasets, group, wrong_penalty=wrong_penalty, num_train=num_train) for wrong_penalty in wrong_penalties}
        if len(datasets) > 1:
            for wrong_penalty in score_data:
                make_thresholds_table(train_data, output_dir, group, wrong_penalty, num_train)
        # Calibration curve plots, where we do collapse across prompts (and datasets)
        if platt:
            collapsed_test_data = apply_platt_scaling(collapse_data_to_model(test_data), platt_params)
        else: # If we're not using platt scaling, we can just use all the data
            collapsed_test_data = collapse_data_to_model(data)
        calibration_curve_plot(collapsed_test_data, output_dir, group_tag)
    else:
        # Make scatter plots for AUROC, calibration error, and accuracy (and return those metrics)
        if 'prob' in group and 'max' in group:
            if platt:
                scatter_data = apply_platt_scaling(data, platt_params)
                ylabel = 'calib-platt'
            else:
                scatter_data = data
                ylabel = 'calib'
            calib_errors, _, _ = acc_scatter_plots(scatter_data, compute_calibration_error, output_dir, ylabel, group) # normal data for scatter plots, average across datasets like for AUROC
            # platt_scaling_plots(test_data, train_data, output_dir, group_tag, dset_tag, num_train)
        # AUROC
        aucs, accs, _ = acc_scatter_plots(data, compute_auroc, output_dir, 'auc', group)

    return score_data, aucs, calib_errors, accs, model_names

def pairwise_group_plots(group_metrics, output_dir, cross_group_name, dset_tag, platt):
    print(f"\nGENERATING PAIRWISE GROUP PLOTS: {cross_group_name}\n")
    # First plot: AUC vs accuracy, but with different colors for each group
    plt.figure()
    texts = []
    for group in sorted(list(group_metrics.keys())): # Colors should be consistent across plots
        score_data, aucs, calib_errors, accs, model_names = group_metrics[group]
        mark_color, marker, line_color = plot_style_for_group(group)
        aucs, accs = np.array(aucs), np.array(accs)
        group_label = group_label_human_readable(group)
        # accuracy is x-axis, auc is y-axis
        plt.scatter(accs, aucs, label=group_label, c=mark_color, marker=marker)
        for i in range(len(model_names)):
            texts.append(plt.text(accs[i], aucs[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7, fontsize='small'))

    plt.legend()
    file_suffix = get_file_suffix(dset_tag, cross_group_name)
    finalize_plot(output_dir, 'acc', 'auc', file_suffix=f"{file_suffix}_side_by_side", texts=texts)
    
    # Second: plots for the averaged group
    _, avg_aucs, avg_calib_errors, avg_accs, model_names = merge_group_metrics(group_metrics)
    model_sizes = [model_size(model) for model in model_names]
    scatter_plot(avg_accs, avg_aucs, output_dir, model_names, 'acc', 'auc',
                 cross_group_name, dset_tag)
    scatter_plot(model_sizes, avg_aucs, output_dir, model_names, 'size', 'auc',
                 cross_group_name, dset_tag)
    if 'prob' in cross_group_name and 'max' in cross_group_name:
        ylabel = 'calib-platt' if platt else 'calib'
        scatter_plot(avg_accs, avg_calib_errors, output_dir, model_names,
                     'acc', ylabel, cross_group_name, dset_tag)
        make_calibration_table(avg_accs, avg_calib_errors, model_names,
                               output_dir, cross_group_name, dset_tag)

def make_all_pairwise_group_plots(group_metrics, output_dir, datasets_to_analyze, collapse_prompts, num_train, platt):
    print(f"\nNow moving to pairwise group plots using groups: {list(group_metrics.keys())}\n")
    dset_tag = dataset_tag(datasets_to_analyze)

    for group1 in group_metrics:
        for group2 in group_metrics:
            conf_type1, prompt_type1, conf_measure1, renorm_status1 = parse_group_name(group1)
            conf_type2, prompt_type2, conf_measure2, renorm_status2 = parse_group_name(group2)
            if group1 > group2 and groups_differ_by_one_component(group1, group2): # We only want to plot pairs that differ by one component. The greater than is so we only do each pair once
                
                # We do pretty different things based on collapse_prompts, based on the types of plots we want. If we haven't already collapsed prompts, we only want to average over prompts. When collapse_prompt=True, we make the score table which has both MSP and Max Logit stats
                if not collapse_prompts and prompt_type1 != prompt_type2: 
                    name_diff = lambda s1,s2: '' if s1 == None == s2 else s1 if s1 == s2 else f'{s1}_vs_{s2}'
                    cross_group_name = f"{name_diff(conf_type1, conf_type2)}_{name_diff(prompt_type1, prompt_type2)}_{name_diff(conf_measure1, conf_measure2)}_{name_diff(renorm_status1, renorm_status2)}"
                    cross_group_name = cross_group_name.replace('_second_prompt_vs_first_prompt','')

                    new_output_dir = os.path.join(output_dir, 'pairwise_group_plots', cross_group_name)
                    pairwise_group_plots({group1: group_metrics[group1], group2: group_metrics[group2]}, new_output_dir, '_' + cross_group_name, dset_tag, platt)
                if collapse_prompts and conf_type1 != conf_type2:
                    print("Making score tables for", group1, "and", group2)
                    # prob is group 1 and logit is group 2 because group 1 > group 2
                    group_tag = f'{conf_measure1}_{renorm_status1}'
                    make_score_table(group_metrics[group1], group_metrics[group2],
                                     output_dir, group_tag, dset_tag, num_train=num_train)
                    make_score_table(group_metrics[group1], group_metrics[group2],
                                     output_dir, group_tag, dset_tag, pct_abstained=True, num_train=num_train)
                    

def make_all_mega_group_plots(group_metrics, output_dir, datasets_to_analyze):
    # Only if collapse_prompts = False, compare MSP to Max Logit, averaged across two prompts
    # Do so for each of (max, margin)
    renorm_status = 'renorm' # Always renormalize probs, but keep this variable for backwards compatibility
    for conf_measure in ('max', 'margin', 'entropy'):
        print(f"\nGENERATING MEGA PLOTS for {datasets_to_analyze}, {conf_measure}, {renorm_status}\n")
        group1a = f"prob,first_prompt,{conf_measure},{renorm_status}"
        group1b = f"prob,second_prompt,{conf_measure},{renorm_status}"
        group2a = f"logit,first_prompt,{conf_measure}"
        group2b = f"logit,second_prompt,{conf_measure}"
        new_group1 = merge_group_metrics({group1a: group_metrics[group1a], group1b: group_metrics[group1b]})
        if conf_measure == 'entropy': # Entropy doesn't have a logit version
            new_group2 = ({}, [], [], [], []) # Empty metrics
        else:
            new_group2 = merge_group_metrics({group2a: group_metrics[group2a], group2b: group_metrics[group2b]})
        new_output_dir = os.path.join(output_dir, 'mega_group_plots', f'{conf_measure}_{renorm_status}')
        dset_tag = dataset_tag(datasets_to_analyze)
        group_tag = f'{conf_measure}_{renorm_status}'
        make_auroc_table(new_group1, new_group2, new_output_dir, group_tag, dset_tag)

def num_train_score_plots(all_num_train_group_metrics, output_dir, dset_tag):
    # make dict of the form num_train_data[group][wrong_penalty][model] = (num_train_list, test_scores, pcts_abstained)
    print("\nGENERATING NUM_TRAIN SCORE PLOTS \n")
    formatted_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], []))))
    for num_train in sorted(all_num_train_group_metrics.keys()):
        group_metrics = all_num_train_group_metrics[num_train]
        for group in group_metrics:
            score_data, aucs, calib_errors, accs, model_names = group_metrics[group]
            for wrong_penalty in score_data:
                _, test_scores, base_test_scores, pcts_abstained = score_data[wrong_penalty]
            
                for model in test_scores:
                    num_train_list, test_scores_list, pcts_abstained_list = formatted_data[group][wrong_penalty][model]
                    num_train_list.append(num_train)
                    test_scores_list.append(test_scores[model])
                    pcts_abstained_list.append(pcts_abstained[model])                    

    for group in formatted_data:
        for wrong_penalty in formatted_data[group]:
            ylabel = 'harsh-score' if wrong_penalty == 2 else 'score'
            score_plot(formatted_data[group][wrong_penalty], output_dir, 'num_train', ylabel, dset_tag, group, xscale='log')
                                
def main():
    # Setup
    if len(sys.argv) < 4:
        print("Usage: python plot_data.py <output_directory> <dataset1,dataset2,...> <collapse_prompts> <num_train> <data_file1> [<data_file2> ...]")
        sys.exit(1)
    output_dir = sys.argv[1]
    datasets_to_analyze = sys.argv[2].split(',')
    if any([dataset not in ('arc', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande') for dataset in datasets_to_analyze]):
        raise Exception(f'Second argument must be a comma-separated subset of [arc, hellaswag, mmlu, truthfulqa, winogrande]. Instead it was:', sys.argv[2])
    if sys.argv[3].lower() == 'true':
        collapse_prompts = True
    elif sys.argv[3].lower() == 'false':
        collapse_prompts = False
    else:
        raise Exception(f'Third argument must be True or False. Instead it was:', sys.argv[3])    
    
    file_paths = sys.argv[4:]
    print(f"\n{'='*70}\nMAKING FIGURES FOR {datasets_to_analyze} WITH collapse_prompts={collapse_prompts}\n{'='*70}\n")
    print(f"Reading from {len(file_paths)} data files\n")
    all_data = gather_all_data(output_dir, datasets_to_analyze, collapse_prompts, file_paths)
                
    make_dataset_plots(all_data, output_dir)
    platt = False # whether to use Platt scaling for the calibration analysis

    if not collapse_prompts:
        num_train_options = [20] 
    else:
        num_train_options = [1,2,5,10,20,50,100,200,500] # For collapse_prompts=True, we want to make a plot that compares the effect of num_train
    all_num_train_group_metrics = dict() # Maps num_train to group_metrics dict
    for num_train in num_train_options:
    # Single group plots, save metrics
        group_metrics = dict()
        for group in all_data:
            print(f"\nGENERATING PLOTS FOR {group} and {datasets_to_analyze}")
            print("Using num_train =", num_train, '\n')
            sanitized_group = sanitize_group_name_for_path(group)
            this_output_dir = os.path.join(output_dir, 'single_group_plots', sanitized_group)
            group_metrics[group] = plots_for_group(all_data[group], this_output_dir, collapse_prompts, group, num_train, platt)
        all_num_train_group_metrics[num_train] = group_metrics

        # Cross group plots
        make_all_pairwise_group_plots(group_metrics, output_dir, datasets_to_analyze, collapse_prompts, num_train, platt)
        if not collapse_prompts: # No need for mega average plots if prompts are already collapsed
            make_all_mega_group_plots(group_metrics, output_dir, datasets_to_analyze)

    # How does num_train affect Q&A with abstention scores?
    if len(all_num_train_group_metrics) > 1:
        dset_tag = dataset_tag(datasets_to_analyze)
        num_train_output_dir = os.path.join(output_dir, 'num_train_plots')
        num_train_score_plots(all_num_train_group_metrics, num_train_output_dir, dset_tag)
    print(f"\nFinished generating plots for {datasets_to_analyze} with collapse_prompts={collapse_prompts} and num_train={num_train_options}\n\n")
                        
if __name__ == "__main__":
    main()
