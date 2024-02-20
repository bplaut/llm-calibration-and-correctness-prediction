import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
import os
from collections import defaultdict
from adjustText import adjust_text
from scipy.stats import linregress
import random
from utils import *

def make_and_sort_legend():
    handles, names = plt.gca().get_legend_handles_labels()
    zipped = zip(handles, names)
    sorted_zipped = sorted(zipped, key=lambda x: (model_series(x[1]), model_size(x[1])))
    sorted_handles, sorted_names = zip(*sorted_zipped)
    plt.legend(handles=sorted_handles, labels=sorted_names, fontsize='small')

def compute_auroc(all_data, output_dir, dataset):
    plt.figure()
    aucs = dict()
    for model, (labels, conf_levels, _) in all_data[dataset].items():
        fpr, tpr, _ = roc_curve(labels, conf_levels)
        roc_auc = make_pct(auc(fpr, tpr))
        aucs[model] = roc_auc
        plt.plot(fpr, tpr, lw=2, label=f'{expand_model_name(model)} (area = {roc_auc:.2f})')

    make_and_sort_legend()
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {dataset}')
    # Make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"roc_curve_{dataset}.pdf")
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve for {dataset} saved --> {output_path}")
    return aucs
    
def finalize_plot(output_dir, xlabel, ylabel, title_suffix='', file_suffix='', texts=[]):
    # Consistent axes
    if xlabel == 'acc':
        plt.xlim([28, 72])
    if ylabel == 'auc':
        plt.ylim([50, 71])
    if ylabel == 'score':
        plt.ylim([-15, 65])
    if ylabel == 'harsh-score':
        plt.ylim([-15, 25])

    adjust_text(texts) # Must do this after setting ylim and xlim

    plt.xlabel(expand_label(xlabel))
    plt.ylabel(expand_label(ylabel))

    # Remove some axes based on the way figs are organized in the paper
    if 'raw' in output_dir:
        plt.ylabel('')
        plt.yticks([])
    if ylabel == 'score':
        plt.xlabel('')
        plt.xticks([])

    # Don't include titles for formatting in paper
    # plt.title(f'{expand_label(ylabel)} vs {expand_label(xlabel)}{title_suffix}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{ylabel}_vs_{xlabel}{file_suffix.replace(' ', '_')}.pdf")
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"{ylabel} vs {xlabel} for {title_suffix} saved --> {output_path}")

def scatter_plot(xs, ys, output_dir, model_names, xlabel, ylabel, dataset='all datasets'):
    plt.figure()
    xs, ys = np.array(xs), np.array(ys)
    group = output_dir[output_dir.rfind('/')+1:]
    mark_color, marker, line_color = plot_style_for_group(group)
    scatter = plt.scatter(xs, ys, c=mark_color, marker=marker)
    texts = []

    for i in range(len(model_names)):
        texts.append(plt.text(xs[i], ys[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7))

    slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
    print("slope, r_value, p_value for", xlabel, ylabel, "is", slope, r_value, p_value)
    plt.plot(xs, intercept + slope * xs, color=line_color, linestyle='-')

    plot_name = 'MSP' if group == 'no_abst_norm_logits' else 'Max Logit' if group == 'no_abst_raw_logits' else group
    plot_name = plot_name if dataset == 'all datasets' else f'{plot_name}, {dataset}'
    file_suffix = f"_{dataset}_{plot_name.replace(' ','_').replace(',','')}"
    finalize_plot(output_dir, xlabel, ylabel, title_suffix=f': {plot_name} (r = {r_value:.2f})', file_suffix=file_suffix, texts=texts)
           
def auc_acc_plots(data, all_aucs, output_dir):
    model_aucs, model_accs = defaultdict(list), defaultdict(list)
    for dataset in all_aucs:
        # Same set of models in each dict, so we can just iterate over one dict
        for model in all_aucs[dataset]:
            model_aucs[model].append(all_aucs[dataset][model])
            (labels, _, _) = data[dataset][model]
            model_accs[model].append(make_pct(np.mean(labels)))

    avg_aucs, avg_accs, model_names = [], [], []
    for model in model_aucs:
        avg_aucs.append(np.mean(model_aucs[model]))
        avg_accs.append(np.mean(model_accs[model]))
        model_names.append(model)

    dataset_name = 'all datasets' if len(all_aucs) > 1 else list(all_aucs.keys())[0]
    model_sizes = [model_size(model) for model in model_names]
    scatter_plot(avg_accs, avg_aucs, output_dir, model_names, 'acc', 'auc', dataset_name)
    scatter_plot(model_sizes, avg_aucs, output_dir, model_names, 'size', 'auc', dataset_name)
    scatter_plot(model_sizes, avg_accs, output_dir, model_names, 'size', 'acc', dataset_name)
    
    return avg_aucs, avg_accs, model_names # We'll use these for the cross-group plots

def compute_score(labels, conf_levels, total_qs, thresh, wrong_penalty=1, normalize=True):
    # Score = num correct - num wrong, with abstaining when confidence < threshold
    score = sum([0 if conf < thresh else (1 if label == 1 else -wrong_penalty) for label, conf in zip(labels, conf_levels)])
    return make_pct(score / total_qs) if normalize else score

def score_plot(data, output_dir, xlabel, ylabel, dataset, thresholds_to_mark=dict(), yscale='linear'):
    plt.figure()
    plt.yscale(yscale)
    # define 10 unique linestyles, using custom patterns after the first 4
    linestyles = ['-', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (0.5,0.5,0.5,0.5,2)),(0, (5, 10)),(0, (5, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1)), (0, (0.5, 0.5)), (0,(1,1,1,3))]

    result_thresholds, result_scores, base_scores = dict(), dict(), dict()
    for (model, xs, ys) in data:
        # Mark the provided threshold if given, else mark the threshold with the best score
        if model in thresholds_to_mark:
            thresh_to_mark = thresholds_to_mark[model]
            thresh_idx = np.where(xs == thresh_to_mark)[0][0] # First zero idx is because np.where returns a tuple, second zero idx is because we only want the first index (although there should only be one)
            score_to_mark = ys[thresh_idx]
        else:
            thresh_to_mark_idx = np.argmax(ys)
            thresh_to_mark = xs[thresh_to_mark_idx]
            score_to_mark = max(ys)
        # zorder determines which objects are on top
        plt.scatter([thresh_to_mark], [score_to_mark], color='black', marker='o', s=20, zorder=3)
        base_score = ys[0] # We added -1 to the front for base score, see plot_score_vs_thresholds
        xs, ys = xs[1:], ys[1:] # Remove the -1 for plotting
        plt.plot(xs, ys, label=f"{expand_model_name(model)}", zorder=2, linestyle=linestyles.pop(0), linewidth=2)
        result_thresholds[model] = thresh_to_mark
        base_scores[model] = base_score
        result_scores[model] = score_to_mark

    make_and_sort_legend()
    plt.legend(handlelength=2.5)
    plot_name = 'MSP' if 'norm' in output_dir else 'Max Logit' if 'raw' in output_dir else 'unknown'
    plot_name = plot_name if dataset == 'all datasets' else f'{plot_name}, {dataset}'
    finalize_plot(output_dir, xlabel, ylabel, title_suffix = f': {plot_name}', file_suffix = f'_{dataset}')
    return result_thresholds, result_scores, base_scores
    
def plot_score_vs_thresholds(data, output_dir, datasets, wrong_penalty=1, thresholds_to_mark=dict()):
    # Inner max is for one model + dataset, middle max is for one dataset, outer max is overall
    max_conf = max([max([max(conf_levels) for _, (_,conf_levels,_) in data[dataset].items()])
                    for dataset in datasets])
    thresholds = np.linspace(0, max_conf, 200) # 200 data points per plot
    thresholds = np.append([-1], thresholds) # The base score is the score when the threshold is 0, but this could cause float issues if confidence is also exactly zero at times. So add -1

    if abs(max_conf - 1) < 0.01: # We're dealing with probabilities: add more points near 1
        thresholds = np.append(thresholds, np.linspace(0.99, 1, 100))
    # Add all keys in thresholds_to_mark to thresholds, and sort
    thresholds = np.sort(np.unique(np.append(thresholds, list(thresholds_to_mark.values()))))

    # For each model and dataset, compute the score for each threshold
    results = defaultdict(lambda: defaultdict(list))        
    for dataset in datasets:
        for model, (labels, conf_levels, total_qs) in data[dataset].items():
            scores  = []
            scores_harsh = []
            for thresh in thresholds:
                score = compute_score(labels, conf_levels, total_qs, thresh, wrong_penalty)
                scores.append(score)
            results[model][dataset] = scores
            
    # Now for each model and threshold, average the scores across datasets
    overall_results = []
    for model in results:
        results_for_model = []
        for i in range(len(thresholds)):
            # Some models might not have results for all datasets (although eventually they should)
            scores_for_thresh = [results[model][dataset][i] for dataset in results[model]]
            avg_score = np.mean(scores_for_thresh)
            results_for_model.append(avg_score)
        overall_results.append((model, thresholds, results_for_model))
            
    dataset_name = 'all datasets' if len(datasets) > 1 else datasets[0]
    ylabel = 'score' if wrong_penalty == 1 else 'harsh-score' if wrong_penalty == 2 else f'Wrong penalty of {wrong_penalty}'
    return score_plot(overall_results, output_dir, 'conf', ylabel, dataset_name, thresholds_to_mark)

def train_and_test_score_plots(test_data, train_data, output_dir, datasets, wrong_penalty=1):
    # Get optimal thresholds for train data, use those to compute scores for test data
    (optimal_train_thresholds, _, _) = plot_score_vs_thresholds(train_data, os.path.join(output_dir, 'train'), datasets, wrong_penalty=wrong_penalty)
    (_, test_scores, base_test_scores) = plot_score_vs_thresholds(test_data, os.path.join(output_dir, 'test'), datasets, wrong_penalty=wrong_penalty, thresholds_to_mark=optimal_train_thresholds)
    return optimal_train_thresholds, test_scores, base_test_scores

def make_auroc_table(msp_group_data, max_logit_group_data, output_dir, dataset=''):
    model_results_msp = make_model_dict(*msp_group_data)
    model_results_max_logit = make_model_dict(*max_logit_group_data)
    rows = []
    # Sort the rows by model series, then by model size
    for model in sorted(model_results_msp.keys(), key=lambda x: (model_series(x), model_size(x))):
        (auc_msp, acc_msp, _) = model_results_msp[model]
        (auc_max_logit, acc_max_logit, _) = model_results_max_logit[model]
        if abs(acc_msp - acc_max_logit) > 0.01:
            print(f"Warning: accuracies for {model} don't match: {acc_msp} vs {acc_max_logit}")
        rows.append([expand_model_name(model), acc_msp, auc_msp, '', auc_max_logit, ''])
    column_names = ['LLM', 'Q\\&A Accuracy', 'AUROC', '$p < 10^{-5}$', 'AUROC', '$p < 10^{-5}$']
    header_row = '& & \\multicolumn{2}{c|}{MSP} & \\multicolumn{2}{c}{Max Logit} \\\\ \n'
    dataset_for_caption = '' if dataset == '' else f' for {format_dataset_name(dataset)}'
    dataset_for_label = '' if dataset == '' else f'{dataset}_'
    make_results_table(column_names, rows, output_dir, caption=f'AUROC results{dataset_for_caption}. AUROC and Q\&A values are percentages, averaged over the two prompts.', label=f'tab:{dataset_for_label}auroc', filename=f'{dataset_for_label}auroc_table.tex', header_row=header_row)

def make_score_table(msp_group_data, max_logit_group_data, output_dir, dataset=''):
    model_results_msp = make_model_dict(*msp_group_data)
    model_results_max_logit = make_model_dict(*max_logit_group_data)
    rows = []
    # Sort the rows by model series, then by model size
    for model in sorted(model_results_msp.keys(), key=lambda x: (model_series(x), model_size(x))):
        (_, _, score_data_msp) = model_results_msp[model]
        (_, _, score_data_max_logit) = model_results_max_logit[model]
        rows.append([expand_model_name(model)])
        for wrong_penalty in score_data_msp:
            (_, score_msp, base_score_msp) = score_data_msp[wrong_penalty]
            (_, score_max_logit, base_score_max_logit) = score_data_max_logit[wrong_penalty]
            if abs(base_score_msp - base_score_max_logit) > 0.01:
                print(f"Warning: base scores for {model} don't match: {base_score_msp} vs {base_score_max_logit}")
            rows[-1].extend([base_score_msp, score_msp, score_max_logit])
    column_names = ['LLM', 'Base LLM', 'MSP', 'Max Logit', 'Base LLM', 'MSP', 'Max Logit']
    header_row = '& \\multicolumn{3}{c|}{Balanced Score} & \\multicolumn{3}{c}{Conservative Score} \\\\ \n'
    dataset_for_caption = '' if dataset == '' else f' for {format_dataset_name(dataset)}'
    dataset_for_label = '' if dataset == '' else f'{dataset}_'
    make_results_table(column_names, rows, output_dir, caption=f'Score results{dataset_for_caption}. All values are percentages. ``Balanced" and ``conservative" correspond to -1 and -2 points per wrong answer, respectively. Correct answers and abstentions are always worth +1 and 0 points, respectively. The total number of points is divided by the total number of questions to obtain the percentages shown in the table.', label=f'tab:{dataset_for_label}score', filename=f'{dataset_for_label}score_table.tex', header_row=header_row)

def percentile_conf_level(data, model, percentiles):
    conf_levels = []
    for dataset in data:
        conf_levels.extend(data[dataset][model][1])
    return np.percentile(conf_levels, percentiles)

def make_percentile_conf_table(data, output_dir, dataset=''):
    rows = []
    percentiles = [10,50,90]
    models = data[list(data.keys())[0]].keys() # All datasets have the same list of models
    for model in sorted(models, key=lambda x: (model_series(x), model_size(x))):
        confs = [make_pct(x) for x in percentile_conf_level(data, model, percentiles)]
        rows.append([expand_model_name(model)] + confs)
    column_names = ['LLM'] + [f'{x}th' for x in percentiles]
    header_row = '& \\multicolumn{3}{c}{Confidence level percentiles} \\\\ \n'
    dataset_for_caption = '' if dataset == '' else f' for {format_dataset_name(dataset)}'
    dataset_for_label = '' if dataset == '' else f'{dataset}_'
    make_results_table(column_names, rows, output_dir, caption=f'Percentile confidence levels{dataset_for_caption}.', label=f'tab:{dataset_for_label}percentile_conf', filename=f'{dataset_for_label}conf_distribution.tex', header_row=header_row)

def make_dataset_table(all_data, output_dir):
    # one row per dataset, one column for avg acc, one for avg MSP auc, one for avg max logit auc
    # all_data[group][dataset][model] = (labels, conf_levels, total_qs)
    dataset_stats = defaultdict(lambda: ([], [], []))
    for group in all_data:
        for dataset in all_data[group]:
            for model in all_data[group][dataset]:
                labels, conf_levels, _ = all_data[group][dataset][model]
                fpr, tpr, _ = roc_curve(labels, conf_levels)
                auc_val = make_pct(auc(fpr, tpr))
                acc_val = make_pct(np.mean(labels))
                dataset_stats[dataset][0].append(acc_val)
                if 'norm' in group:
                    dataset_stats[dataset][1].append(auc_val)
                elif 'raw' in group:
                    dataset_stats[dataset][2].append(auc_val)

    rows = []
    for dataset in sorted(dataset_stats.keys()):
        accs, msp_aucs, max_logit_aucs = dataset_stats[dataset]
        rows.append([format_dataset_name(dataset), np.mean(accs), np.mean(msp_aucs), np.mean(max_logit_aucs)])
    column_names = ['Dataset', 'Q\\&A Accuracy', 'MSP AUROC', 'Max Logit AUROC']
    make_results_table(column_names, rows, output_dir, caption='Average Q\\&A accuracy and AUROCs per dataset. All values are percentages, averaged over the then models and two prompts.', label='tab:dataset', filename='dataset.tex')
    
def make_results_table(column_names, rows, output_dir, caption='', label='', filename='table.tex', header_row=''):
    filename = os.path.join(output_dir, filename)
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, 'w') as f:
        f.write('\\begin{table*}\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular}{' + 'c|' * (len(column_names) - 1) + 'c}\n')
        f.write(header_row)
        f.write(' & '.join(column_names) + '\\\\ \\hline\n')
        for row in rows:
            # round floats to 1 decimal place, but if it's -0.0, make it 0.0
            row = [str(round(x, 1)) if isinstance(x, float) else str(x) for x in row]
            row = [x.replace('-0.0', '0.0') for x in row]
            f.write(' & '.join(row) + '\\\\\n')
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write(f'\\caption{{{caption}}}\n')
        f.write(f'\\label{{{label}}}\n')
        f.write('\\end{table*}\n')
    print("Results table saved -->", filename)

def plots_for_group(data, output_dir):
    # Split into train and test. We don't have to shuffle, since question order is already randomized
    train_data, test_data = defaultdict(dict), defaultdict(dict)
    for dataset in data:
        for model in data[dataset]:
            labels, conf_levels, total_qs = data[dataset][model]
            # Shuffle labels and conf_levels together
            random.seed(2549900867) # Everyone should have the same split
            combined = list(zip(labels, conf_levels))
            random.shuffle(combined)
            labels, conf_levels = zip(*combined)
            n = len(labels)
            train_data[dataset][model] = (labels[:n//2], conf_levels[:n//2], total_qs/2)
            test_data[dataset][model] = (labels[n//2:], conf_levels[n//2:], total_qs/2)
            
    # ROC plots (also collecting auc data)
    all_aucs = dict()
    for dataset in data:
        all_aucs[dataset] = compute_auroc(data, output_dir, dataset)

    # Main plots
    datasets = list(data.keys())
    plot_score_vs_thresholds(data, output_dir, datasets, wrong_penalty=1)
    plot_score_vs_thresholds(data, output_dir, datasets, wrong_penalty=2)
    # Maps each wrong_penalty to (train_thresholds, test_scores, base_test_scores)
    score_data = {1: train_and_test_score_plots(test_data, train_data, output_dir, datasets, wrong_penalty=1), 2: train_and_test_score_plots(test_data, train_data, output_dir, datasets, wrong_penalty=2)}
    make_percentile_conf_table(data, output_dir)

    aucs, accs, model_names = auc_acc_plots(data, all_aucs, output_dir)
    return score_data, aucs, accs, model_names

def merge_groups(group_data):
    # Merge to a single "group" based on the means across groups
    all_auc_acc_data = defaultdict(lambda: ([], []))
    all_score_data = defaultdict(lambda: (defaultdict(list), defaultdict(list), defaultdict(list)))
    for group in group_data:
        (score_data, aucs, accs, model_names) = group_data[group]
        for i, model_name in enumerate(model_names):
            # Collect the auc and acc for this model in each group into a list
            all_auc_acc_data[model_name][0].append(aucs[i])
            all_auc_acc_data[model_name][1].append(accs[i])
            
        for wrong_penalty in score_data:
            # Same idea here, except these are each dicts with model name as the key
            (thresholds, our_scores, base_scores) = score_data[wrong_penalty]
            for model in thresholds:
                all_score_data[wrong_penalty][0][model].append(thresholds[model])
                all_score_data[wrong_penalty][1][model].append(our_scores[model])
                all_score_data[wrong_penalty][2][model].append(base_scores[model])

    plt.figure()
    avg_aucs, avg_accs, model_names = [], [], []
    for model_name, (aucs, accs) in all_auc_acc_data.items():
        avg_aucs.append(np.mean(aucs))
        avg_accs.append(np.mean(accs))
        model_names.append(model_name)

    new_score_data = defaultdict(lambda: (dict(), dict(), dict()))
    for wrong_penalty, (thresholds, our_scores, base_scores) in all_score_data.items():
        for model in thresholds:
            thresh_list, our_scores_list, base_scores_list = thresholds[model], our_scores[model], base_scores[model]
            new_thresh, new_our_score, new_base_score = np.mean(thresh_list), np.mean(our_scores_list), np.mean(base_scores_list)
            new_score_data[wrong_penalty][0][model] = new_thresh
            new_score_data[wrong_penalty][1][model] = new_our_score
            new_score_data[wrong_penalty][2][model] = new_base_score
    return new_score_data, avg_aucs, avg_accs, model_names

def make_model_dict(score_data, aucs, accs, model_names):
    # Change the dict structure so that the model is the key
    model_results = dict()
    for i, model_name in enumerate(model_names):
        model_score_data = dict()
        for wrong_penalty in score_data:
            thresholds, our_scores, base_scores = score_data[wrong_penalty]
            thresh, our_score, base_score = thresholds[model_name], our_scores[model_name], base_scores[model_name]
            model_score_data[wrong_penalty] = (thresh, our_score, base_score)
        model_results[model_name] = (aucs[i], accs[i], model_score_data)
    return model_results

def cross_group_plots(group_data, output_dir):
    print(f"\nGENERATING CROSS GROUP PLOTS: {list(group_data.keys())}\n")
    # First plot: AUC vs accuracy, but with different colors for each group
    plt.figure()
    texts = []
    for group in sorted(list(group_data.keys())): # Colors should be consistent across plots
        score_data, aucs, accs, model_names = group_data[group]
        mark_color, marker, line_color = plot_style_for_group(group)
        aucs, accs = np.array(aucs), np.array(accs)
        logit_type, prompt = group_label(group)
        # accuracy is x-axis, auc is y-axis
        plt.scatter(accs, aucs, label=logit_type+prompt, c=mark_color, marker=marker)
        for i in range(len(model_names)):
            texts.append(plt.text(accs[i], aucs[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7))

    file_suffix = '-' + '-'.join(group_data.keys())
    if 'prompt' in file_suffix:
        logit_type, _ = group_label(list(group_data.keys())[0])
        title_suffix = ': ' + logit_type + ', prompt comparison'
    else:
        title_suffix = ': ' + ' and '.join([group_label(group)[0] for group in group_data])
    plt.legend(loc='lower right')
    finalize_plot(output_dir, 'acc', 'auc', file_suffix=file_suffix, title_suffix=title_suffix, texts=texts)
    
    # Second plot: AUC vs accuracy, averaged across groups
    score_data, avg_aucs, avg_accs, model_names = merge_groups(group_data)
    model_sizes = [model_size(model) for model in model_names]
    scatter_plot(avg_accs, avg_aucs, output_dir, model_names, 'acc', 'auc')
    scatter_plot(model_sizes, avg_aucs, output_dir, model_names, 'size', 'auc')
    scatter_plot(model_sizes, avg_accs, output_dir, model_names, 'size', 'acc')
    
def main():
    # Setup
    if len(sys.argv) < 6:
        print("Usage: python plot_data.py <output_directory> <incl_unparseable> <collapse_prompts> <dataset1,dataset2,...> <data_file1> [<data_file2> ...]")
        sys.exit(1)
    output_dir = sys.argv[1]
    incl_unparseable = (False if sys.argv[2].lower() == 'false' else True if sys.argv[2].lower() == 'true' else None) # On questions where the model produced an unparseable answer, do we include it and count it as wrong, or discard it?
    collapse_prompts = (False if sys.argv[3].lower() == 'false' else True if sys.argv[3].lower() == 'true' else None) # Do we collapse the two prompts into a single group with 12k questions per dataset?
    
    if incl_unparseable is None:
        raise Exception('Second argument incl_unparseable must be a boolean (True or False). Instead it was:', sys.argv[2])
    if collapse_prompts is None:
        raise Exception('Third argument collapse_prompts must be a boolean (True or False). Instead it was:', sys.argv[3])
    file_paths = sys.argv[5:]
    print(f"Reading from {len(file_paths)} files...")
    datasets_to_analyze = sys.argv[4].split(',')
    if any([dataset not in ('arc', 'hellaswag', 'mmlu', 'piqa', 'truthfulqa', 'winogrande') for dataset in datasets_to_analyze]):
        raise Exception(f'Third argument must be a comma-separated list of datasets')
    if 'all' in datasets_to_analyze:
        datasets_to_analyze = ['arc', 'hellaswag', 'mmlu', 'piqa', 'truthfulqa', 'winogrande']

    # Data aggregation. We want all_data[group][dataset][model] = (labels, conf_levels, total_qs)
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], 0))))
    for file_path in file_paths:
        dataset, model, group = parse_file_name(os.path.basename(file_path), collapse_prompts)
        if dataset in datasets_to_analyze:
            labels, conf_levels, total_qs = parse_data(file_path, incl_unparseable)
            old_labels, old_conf_levels, old_total_qs = all_data[group][dataset][model]
            all_data[group][dataset][model] = (np.concatenate([old_labels, labels]), np.concatenate([old_conf_levels, conf_levels]), old_total_qs + total_qs)

    make_dataset_table(all_data, output_dir)

    # Single group plots
    group_data = dict()
    for group in all_data:
        print(f"\nGENERATING PLOTS FOR {group} and {datasets_to_analyze}\n")
        group_data[group] = plots_for_group(all_data[group], os.path.join(output_dir, group))

    # Cross-group plots
    for group1 in group_data:
        for group2 in group_data:
            if group1 > group2: # greater because we only need to do each pair once
                (abst_type_1, logit_type_1, prompt_type_1) = parse_group_name(group1, collapse_prompts)
                (abst_type_2, logit_type_2, prompt_type_2) = parse_group_name(group2, collapse_prompts)
                data = {group1: group_data[group1], group2: group_data[group2]}
                # Only compare pairs of groups which differ by exactly 1 component
                if abst_type_1 == abst_type_2 and (logit_type_1 == logit_type_2 or prompt_type_1 == prompt_type_2):
                    bottom_dir = f'{abst_type_1}_{logit_type_1}' if logit_type_1 == logit_type_2 else f'{abst_type_1}_{prompt_type_1}'
                    new_output_dir = os.path.join(output_dir, 'cross_group_plots', bottom_dir)
                    cross_group_plots(data, new_output_dir)
                    # For tables, need MSP data as first arg, Max Logit data as second arg
                    if logit_type_1 != logit_type_2:
                        msp_group = group_data[group1 if 'norm' in logit_type_1 else group2]
                        max_logit_group = group_data[group1 if 'raw' in logit_type_1 else group2]
                        dset = '' if len(datasets_to_analyze) > 1 else datasets_to_analyze[0]
                        make_auroc_table(msp_group, max_logit_group, new_output_dir, dataset=dset)
                        make_score_table(msp_group, max_logit_group, new_output_dir, dataset=dset)
                        
    # Finally, compare normed vs raw logits, averaged over the two prompts
    try:
        group1a = 'no_abst_norm_logits_first_prompt'
        group1b = 'no_abst_norm_logits_second_prompt'
        new_group1 = merge_groups({group1a: group_data[group1a], group1b: group_data[group1b]})
        group2a = 'no_abst_raw_logits_first_prompt'
        group2b = 'no_abst_raw_logits_second_prompt'
        new_group2 = merge_groups({group2a: group_data[group2a], group2b: group_data[group2b]})
        merged_groups = {'no_abst_norm_logits': new_group1, 'no_abst_raw_logits': new_group2}
        new_output_dir = os.path.join(output_dir, 'cross_group_plots', 'no_abst_all')
        cross_group_plots(merged_groups, new_output_dir)
        dset = '' if len(datasets_to_analyze) > 1 else datasets_to_analyze[0]
        make_auroc_table(new_group1, new_group2, new_output_dir, dset)
        make_score_table(new_group1, new_group2, new_output_dir, dset)
    except KeyError:
        print("\nCouldn't find the right groups for the overall average plot, skipping.\n")

if __name__ == "__main__":
    main()
