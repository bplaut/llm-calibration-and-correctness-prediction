import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
import os
import math
from collections import defaultdict
from scipy.optimize import minimize
from scipy.special import expit
from utils import *
import random

############
# PLOTTING STYLES
#############

# We want to assign a style to each model globally, even when some models are missing from some groups (i.e., OpenAI models only have MSP, not Max Logit)
linestyles = ['-', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (0.5,0.5,0.5,0.5,2)),(0, (5, 10)),(0, (5.5, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1)), (0, (0.25,0.25)), (0, (5,0.5,0.5,5)), (0, (0.5, 0.5)), (0,(1,1,1,3.5)), (0, (0.5,0.5,0.5,2)), (0,(2,1,2,2)), (0,(0.5,0.5,5,5))]
colors = ['pink', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 'teal', 'slategray', 'indigo', '#bcbd22', 'black', '#17becf', 'black']
style_per_model = dict()
# set all fonts to serif
plt.rcParams['font.family'] = 'serif'

def plot_style_for_group(group):
    """Generate plot style for group format"""
    if 'first_prompt' in group:
        marker = 's'
        base_color = 'teal'
        line_color = 'black'
    elif 'second_prompt' in group:
        marker = '^'
        base_color = 'gold'
        line_color = 'black'
    elif 'prob' in group:
        marker = 'o' 
        base_color = '#1f77b4'  # blue
        line_color = 'tab:red'
    elif 'logit' in group:
        marker = 'D'
        base_color = 'mediumpurple'
        line_color = 'tab:orange'
    else:
        raise ValueError(f"Unexpected group format: {group}")
    
    return base_color, marker, line_color

###############
# GENERAL PLOTTING FUNCTIONS
###############

def make_and_sort_legend():
    handles, names = plt.gca().get_legend_handles_labels()
    zipped = zip(handles, names)
    sorted_zipped = sorted(zipped, key=lambda x: (model_series(x[1]), model_size(x[1])))
    sorted_handles, sorted_names = zip(*sorted_zipped)
    plt.legend(handles=sorted_handles, labels=sorted_names, fontsize='small', bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, handlelength=2.5)

def finalize_plot(output_dir, xlabel, ylabel, file_suffix='', texts=[]):
    # Consistent axes
    if xlabel == 'acc':
        plt.xlim([28, 89])
    if ylabel == 'auc':
        plt.ylim([48, 88])
    if ylabel in ('score', 'harsh-score'):
        plt.ylim([-25, 78])
    if ylabel in ['calib', 'calib-platt']:
        plt.ylim([0, 43])

    adjust_text(texts) # Must do this after setting ylim and xlim

    plt.xlabel(expand_label(xlabel))
    plt.ylabel(expand_label(ylabel))

    # Remove some axes based on the way score figs are organized in the paper
    if 'logit' in file_suffix:
        plt.ylabel('')
        plt.yticks([])
    if ylabel == 'score':
        plt.xlabel('')
        plt.xticks([])

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{xlabel}_vs_{ylabel}{file_suffix}.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"* {ylabel} vs {xlabel} saved --> {output_path}")

def scatter_plot(xs, ys, output_dir, model_names, xlabel, ylabel, group, dset_tag='', num_train=None):
    plt.figure()
    # If we're plotting size, exclude GPT models (because we don't know their size)
    if xlabel == 'size':
        xs = [xs[i] for i in range(len(xs)) if 'gpt' not in model_names[i].lower()]
        ys = [ys[i] for i in range(len(ys)) if 'gpt' not in model_names[i].lower()]
        model_names = [name for name in model_names if 'gpt' not in name.lower()]
    xs, ys = np.array(xs), np.array(ys)
    mark_color, marker, line_color = plot_style_for_group(group)
    scatter = plt.scatter(xs, ys, c=mark_color, marker=marker)
    texts = []

    for i in range(len(model_names)):
        texts.append(plt.text(xs[i], ys[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7))

    try:
        slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
        r2_value = r_value**2
        print("slope, r2_value, p_value for", xlabel, ylabel, "is", slope, r2_value, p_value)
        plt.plot(xs, intercept + slope * xs, color=line_color, linestyle='-')

        file_suffix = get_file_suffix(dset_tag, group, num_train)
        file_suffix += f"_r2_{r2_value:.3f}_p_{p_value:.5f}"
        finalize_plot(output_dir, xlabel, ylabel, file_suffix=file_suffix, texts=texts)
    except ValueError as e:
        print("ValueError when running linear regression on", xlabel, ylabel, ":", e)

def acc_scatter_plots(data, metric_fn, output_dir, ylabel, group, **metric_kwargs):
    """
    Generalized function to create accuracy vs metric scatter plots.
    metric_fn should take (labels, conf_levels) and possibly **metric_kwargs and return a metric value.
    """
    model_metrics, model_accs = defaultdict(list), defaultdict(list)
    
    # Compute metric for each dataset-model pair
    for dataset in data:
        for model in data[dataset]:
            labels, conf_levels, _ = data[dataset][model]
            # Compute the metric using the provided function
            metric_value = metric_fn(labels, conf_levels, **metric_kwargs)
            model_metrics[model].append(metric_value)
            model_accs[model].append(make_pct(np.mean(labels)))
    
    # Average across datasets for each model
    avg_metrics, avg_accs, model_names = [], [], []
    for model in sort_models(model_metrics.keys()):
        avg_metrics.append(np.mean(model_metrics[model]))
        avg_accs.append(np.mean(model_accs[model]))
        model_names.append(model)
    
    # Create scatter plots
    dset_tag = dataset_tag(list(data.keys()))
    model_sizes = [model_size(model) for model in model_names]
    scatter_plot(avg_accs, avg_metrics, output_dir, model_names, 'acc', ylabel, group, dset_tag)
    scatter_plot(model_sizes, avg_metrics, output_dir, model_names, 'size', ylabel, group, dset_tag)
    
    return avg_metrics, avg_accs, model_names

def make_table(num_cols, rows, output_dir, caption='', label='', filename='table.tex', header='', precision=1, use_siunitx=False):
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write('\\begin{table*}[h]\n')
        f.write(f'\\caption{{{caption}}}\n')
        f.write(f'\\label{{{label}}}\n')
        f.write('\\centering\n')
        if use_siunitx:
            f.write('\\begin{tabular}{' + 'l' + ' S[table-format=3.1]' * (num_cols - 1) + '}\n') # S[table-format=3.1] is an alignment column type from the siunitx package
        else:
            f.write('\\begin{tabular}{' + 'l' + 'c' * (num_cols - 1) + '}\n')
        f.write('\\toprule\n')
        f.write(header)
        for row in rows:
            # round floats to {precision} decimal place, but if it's -0.0, make it 0.0
            row = [str(round(x, precision)) if isinstance(x, float) else str(x) for x in row]
            row = [x.replace('-0.0', '0.0') for x in row]
            f.write(' & '.join(row) + '\\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table*}\n')
    print("* Results table saved -->", filename)

###########
# AUROC STUFF
###########

def compute_auroc(labels, conf_levels):
    """Compute AUC for a single dataset-model pair"""
    fpr, tpr, _ = roc_curve(labels, conf_levels)
    return make_pct(auc(fpr, tpr))

def make_auroc_table(msp_group_metrics, max_logit_group_metrics, output_dir, tag='', dataset=''):
    model_results_msp = make_model_dict(*msp_group_metrics)
    model_results_max_logit = make_model_dict(*max_logit_group_metrics)
    rows = []
    # Sort the rows by model series, then by model size. Also put gpt models at the end
    for model in sort_models(model_results_msp.keys()):
        # Default is ('', '', '', {}) if we don't have results for that model. This can happen with e.g. GPT models where we only have MSP results, not raw logits.
        (auc_msp, _, acc_msp, _) = model_results_msp.get(model, ('', '', 0, {}))
        (auc_max_logit, _, acc_max_logit, _) = model_results_max_logit.get(model, ('', '', 0, {}))
        if abs(acc_msp - acc_max_logit) > 0.01 and 'gpt' not in model and 'entropy' not in tag: # entropy only has MSP, so we don't compare accuracies
            print(f"Warning: accuracies for {model} don't match: {acc_msp} vs {acc_max_logit}")
        rows.append([expand_model_name(model), acc_msp, auc_msp, '2/2', auc_max_logit, '2/2'])
    column_names = ['LLM', 'Q\\&A Accuracy', 'AUROC', '$p < 10^{-4}$', 'AUROC', '$p < 10^{-4}$']
    header = ('& & \\multicolumn{2}{c}{MSP} & \\multicolumn{2}{c}{Max Logit} \\\\ \n'
              + ' & '.join(column_names) + ' \\\\ \n'
              + '\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \n')
    caption = 'AUROC results for %s. See Table~\\ref{tab:auroc} for an explanation of the $p$-values.' % format_dataset_name(dataset)
    dset_tag = dataset_tag(dataset)
    file_suffix = get_file_suffix(dset_tag, tag)
    make_table(len(column_names), rows, output_dir, caption=caption, label=f'tab:auroc{file_suffix}', filename=f'auroc{file_suffix}_table.tex', header=header)

def make_dataset_plots(all_data, output_dir):
    # one entry per dataset, containing avg acc, avg MSP auc, and avg max logit auc
    # all_data[group][dataset][model] = (labels, conf_levels, total_qs)
    dataset_stats = defaultdict(lambda: ([], [], [], []))
    # First collect all the data for each dataset
    for group in all_data:
        conf_type, _, conf_measure, renorm_status = parse_group_name(group)
        for dataset in all_data[group]:
            for model in all_data[group][dataset]:
                labels, conf_levels, _ = all_data[group][dataset][model]
                fpr, tpr, _ = roc_curve(labels, conf_levels)
                auc_val = make_pct(auc(fpr, tpr))
                acc_val = make_pct(np.mean(labels))
                dataset_stats[dataset][0].append(acc_val)
                if (conf_type, conf_measure, renorm_status) == ('prob', 'max', 'renorm'):
                    dataset_stats[dataset][1].append(auc_val)
                    calib_error = make_pct(compute_calibration_error(labels, conf_levels))
                    dataset_stats[dataset][3].append(calib_error)

                elif (conf_type, conf_measure, renorm_status) == ('logit', 'max', None):
                    dataset_stats[dataset][2].append(auc_val)

    if len(dataset_stats) < 2:
        # No point in making a dataset comparison for less than 2 datasets
        print("Not enough datasets to make a comparison. Skipping dataset plots.")
        return
    
    # Otherwise, average the data for each dataset
    for dataset in dataset_stats:
        accs, msp_aucs, max_logit_aucs, calib_errors = dataset_stats[dataset]
        if accs:  # Only process if we have data
            dataset_stats[dataset] = (np.mean(accs), np.mean(msp_aucs), np.mean(max_logit_aucs), np.mean(calib_errors))
        
    # Make table
    rows = []
    for dataset in sorted(dataset_stats.keys()):
        avg_acc, msp_auc, max_logit_auc, calib_error = dataset_stats[dataset]
        rows.append([format_dataset_name(dataset), avg_acc, msp_auc, max_logit_auc, calib_error])
    column_names = ['Dataset', 'Q\\&A Accuracy', 'MSP AUROC', 'Max Logit AUROC', 'Calibration Error (x100)']
    filename = 'dataset_table.tex'
    header = (' & ' + ' & '.join(column_names[1:]) + ' \\\\ \n'
              '\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \\cmidrule(lr){3-3} \\cmidrule(lr){4-4} \\cmidrule(lr){5-5}\n')
    make_table(len(column_names), rows, output_dir, caption='Average Q\\&A accuracy, AUROCs, and calibration error per dataset. All values are percentages, averaged over the then models and two prompts.', label='tab:dataset', filename=filename, header=header)

    # Make bar graph. Three segments on the x-axis: Q&A accuracy, MSP AUROC, Max Logit AUROC. Within each segment, one bar per dataset. So there should be three segments, each with 5 bars
    labels = ['Q&A Accuracy', 'MSP AUROC', 'Max Logit AUROC', 'Calibration Error']
    n_groups = len(labels)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15

    for i, dataset in enumerate(sorted(dataset_stats.keys())):
        means = list(dataset_stats[dataset])
        plt.bar(index + i*bar_width, means, bar_width, label=format_dataset_name(dataset))

    plt.ylabel('Percentage')
    plt.xticks(index + 2*bar_width, labels)
    plt.ylim([0, 90])
    plt.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'dataset_bar.pdf')
    plt.savefig(filepath)
    print("* Dataset bar graph saved -->", filepath)
    plt.close()

#################
# Q&A WITH ABSTENTION
#################

def compute_abstention_score(labels, conf_levels, total_qs, thresh, wrong_penalty=1, normalize=True):
    # Score = num correct - num wrong, with abstaining when confidence < threshold
    score = sum([0 if conf < thresh else (1 if label == 1 else -wrong_penalty) for label, conf in zip(labels, conf_levels)])
    return make_pct(score / total_qs) if normalize else score

def score_plot(data, output_dir, xlabel, ylabel, dataset, group, thresholds_to_use=dict(), xscale='linear', yscale='linear', num_train=None):
    # data must have the form data[model] = (xs, ys, pct_abstained_all_threshes)
    plt.figure()
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xlabel == 'num_train':
        (xs, _, _) = next(iter(data.values())) # arbitrary data point, just need xs
        plt.tick_params(axis='x', which='minor', bottom=False) # Remove minor ticks
        plt.xticks(xs, xs) # Major xticks at num_train values labeled by the actual value

    result_thresholds, result_scores, base_scores, pcts_abstained = dict(), dict(), dict(), dict()
    for model in sort_models(data.keys()):
        (xs, ys, pct_abstained_all_threshes) = data[model]
        if model not in style_per_model:
            style_per_model[model] = (linestyles.pop(0), colors.pop(0))
        (linestyle, color) = style_per_model[model]
        # If thresholds_to_use is provided, mark the threshold for that model
        if model in thresholds_to_use:
            chosen_thresh = thresholds_to_use[model]
            chosen_thresh_idx = np.where(xs == chosen_thresh)[0][0] # First 0 idx is because np.where returns a tuple, second 0 idx is because we only want the 1st index (there should only be 1)
            chosen_score = ys[chosen_thresh_idx]
            # zorder determines which objects are on top
            plt.scatter([chosen_thresh], [chosen_score], color='black', marker='o', s=20, zorder=3)
        else: # Use the optimal threshold. This is what we'll do during training
            chosen_thresh_idx = np.argmax(ys)
            chosen_thresh = xs[chosen_thresh_idx]
            chosen_score = ys[chosen_thresh_idx]
        pct_abstained = pct_abstained_all_threshes[chosen_thresh_idx]
        result_thresholds[model] = chosen_thresh
        base_scores[model] = ys[0] # We added float(-inf) to the front for base score, see plot_score_vs_threshold
        result_scores[model] = chosen_score
        pcts_abstained[model] = pct_abstained

        plt.plot(xs, ys, label=f"{expand_model_name(model)}", zorder=2, linestyle=linestyle, linewidth=2, color=color)

    if 'prob' in group and ylabel == 'score':
        # Only make legend for upper left plot in paper
        make_and_sort_legend()
    
    file_suffix = get_file_suffix(dataset, group, num_train)
    finalize_plot(output_dir, xlabel, ylabel, file_suffix = file_suffix)
    return result_thresholds, result_scores, base_scores, pcts_abstained
    
def plot_score_vs_thresholds(data, output_dir, datasets, group, wrong_penalty=1, thresholds_to_use=dict(), num_train=None):
    # Inner max is for one model + dataset, middle max is for one dataset, outer max is overall
    max_conf = max([max([max(conf_levels) for _, (_,conf_levels,_) in data[dataset].items()])
                    for dataset in datasets])
    min_conf = min([min([min(conf_levels) for _, (_,conf_levels,_) in data[dataset].items()])
                    for dataset in datasets])
    thresholds = np.linspace(min_conf - 0.0001, max_conf, 200) # 200 data points per plot. Add -0.0001 to avoid floating point issues with confidence levels that are exactly 0

    if abs(max_conf - 1) < 0.01: # We're dealing with probabilities: add more points near 1
        thresholds = np.append(thresholds, np.linspace(0.99, 1, 100))
    # Add all keys in thresholds_to_use to thresholds, and sort
    thresholds = np.sort(np.unique(np.append(thresholds, list(thresholds_to_use.values()))))

    # For each model and dataset, compute the score and pct abstained for each threshold
    all_scores = defaultdict(lambda: defaultdict(list))
    all_pcts_abstained = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        for model, (labels, conf_levels, total_qs) in data[dataset].items():
            scores  = []
            pcts_abstained = []
            for thresh in thresholds:
                scores.append(compute_abstention_score(labels, conf_levels, total_qs, thresh, wrong_penalty))
                pcts_abstained.append(make_pct(np.mean([1 if conf < thresh else 0 for conf in conf_levels])))
            all_scores[model][dataset] = scores
            all_pcts_abstained[model][dataset] = pcts_abstained
            
    # Now for each model and threshold, average the score and the pct abstained across datasets
    overall_results = dict()
    for model in all_scores:
        scores_for_model = []
        pcts_abstained_for_model = []
        for i in range(len(thresholds)):
            # Some models might not have results for all datasets (although eventually they should)
            scores_for_thresh = [all_scores[model][dataset][i] for dataset in all_scores[model]]
            avg_score = np.mean(scores_for_thresh)
            pct_abstained_per_dataset = np.mean([all_pcts_abstained[model][dataset][i] for dataset in all_pcts_abstained[model]])
            avg_pct_abstained = np.mean(pct_abstained_per_dataset)
            scores_for_model.append(avg_score)
            pcts_abstained_for_model.append(avg_pct_abstained)
        overall_results[model] = (thresholds, scores_for_model, pcts_abstained_for_model)
            
    dset_tag = dataset_tag(datasets)
    ylabel = 'score' if wrong_penalty == 1 else 'harsh-score' if wrong_penalty == 2 else f'Wrong penalty of {wrong_penalty}'
    return score_plot(overall_results, output_dir, 'conf', ylabel, dset_tag, group, thresholds_to_use, num_train=num_train)

def train_and_test_score_plots(test_data, train_data, output_dir, datasets, group, wrong_penalty=1, num_train=None):
    # Get optimal thresholds for train data, use those to compute scores for test data
    (optimal_train_thresholds, _, _, _) = plot_score_vs_thresholds(train_data, os.path.join(output_dir, 'train'), datasets, group, wrong_penalty=wrong_penalty, num_train=num_train)
    (_, test_scores, base_test_scores, pcts_abstained) = plot_score_vs_thresholds(test_data, os.path.join(output_dir, 'test'), datasets, group, wrong_penalty=wrong_penalty, thresholds_to_use=optimal_train_thresholds, num_train=num_train)
    return optimal_train_thresholds, test_scores, base_test_scores, pcts_abstained

def make_thresholds_table(train_data, output_dir, group, wrong_penalty, num_train):
    datasets = sorted(list(train_data.keys()))
    all_thresholds = defaultdict(dict)  # all_thresholds[model][dataset] = threshold
    
    # Get optimal threshold for each dataset separately
    for dataset in datasets:
        single_dataset_data = {dataset: train_data[dataset]}
        thresholds, _, _, _ = plot_score_vs_thresholds(single_dataset_data, os.path.join(output_dir, 'per_dataset_thresholds', dataset), [dataset], group, wrong_penalty=wrong_penalty, num_train=num_train)
        
        for model in thresholds:
            all_thresholds[model][dataset] = thresholds[model]
    
    # Create table rows
    rows = []
    models = sort_models(all_thresholds.keys())
    
    for model in models:
        row = [expand_model_name(model)]
        for dataset in datasets:
            threshold = all_thresholds[model].get(dataset, '')
            precision = 3 if 'prob' in group else 1
            row.append(f"{threshold:.{precision}f}")
        rows.append(row)
    
    # Create table
    column_names = ['LLM'] + [format_dataset_name(d) for d in datasets]
    header = ' & '.join(column_names) + ' \\\\ \n'
    header += '\\cmidrule(lr){1-1} ' + ' '.join([f'\\cmidrule(lr){{{i+2}-{i+2}}}' for i in range(len(datasets))]) + '\n'
    file_suffix = get_file_suffix('', group, num_train)
    filename = f'thresholds{file_suffix}_w{wrong_penalty}_table.tex'
    penalty_str = 'conservative' if wrong_penalty == 2 else 'balanced'
    caption = f'Thresholds for each model on each dataset ({penalty_str} scoring, {group_label_human_readable(group)[0]}). '
    label = f'tab:thresholds{file_suffix}_w{wrong_penalty}'
    make_table(len(column_names), rows, output_dir, caption=caption, label=label, filename=filename, header=header, use_siunitx=True)

def make_score_table(msp_group_metrics, max_logit_group_metrics, output_dir, group, dset_tag, pct_abstained=False, num_train=None):
    # If pct_abstained=True, we'll write the pct_abstained instead of the score
    model_results_msp = make_model_dict(*msp_group_metrics)
    model_results_ml = make_model_dict(*max_logit_group_metrics)
    rows = []
    for model in sort_models(model_results_msp.keys()):
        # Default is (0, 0, 0, {}) if we don't have results for that model
        (_, _, _, score_data_msp) = model_results_msp.get(model, (0, 0, 0, {}))
        (_, _, _, score_data_ml) = model_results_ml.get(model, (0, 0, 0, {}))
        rows.append([expand_model_name(model)])
        for wrong_penalty in score_data_msp:
            # '' is the default if we're missing data for a model (e.g., we're missing raw logits for OpenAI models)
            (_, score_msp, base_score_msp, pct_abstained_msp) = score_data_msp.get(wrong_penalty, (0, '', 0, ''))
            (_, score_ml, base_score_ml, pct_abstained_ml) = score_data_ml.get(wrong_penalty, (0, '', 0, ''))
            if abs(base_score_msp - base_score_ml) > 0.01 and 'gpt' not in model:
                print(f"Warning: base scores for {model} don't match: {base_score_msp} vs {base_score_ml}\n")
            if pct_abstained:
                rows[-1].extend([0, pct_abstained_msp, pct_abstained_ml])
            else:
                rows[-1].extend([base_score_msp, score_msp, score_ml])
    column_names = ['LLM', 'No abstain', 'MSP', 'Max Logit', 'No abstain', 'MSP', 'Max Logit']
    formatted_column_names = ["\\multicolumn{1}{c}{" + name + "}" for name in column_names] # Wrap in multicolumn to make siuntx ignore them when aligning numerical values   
    header = ('& \\multicolumn{3}{c}{Balanced} & \\multicolumn{3}{c}{Conservative} \\\\ \n'
              + ' & '.join(formatted_column_names) + ' \\\\ \n'
              + '\\cmidrule(lr){1-1}\\cmidrule(lr){2-4}\\cmidrule(lr){5-7} \n')
    caption = ('Q\\&A with abstention results for %s. See Table~\\ref{tab:score} for an explanation of the scoring scheme.' if not pct_abstained else 'Frequency of abstention on %s in the Section~\\ref{sec:abstain} experiments.') % format_dataset_name(dset_tag)
    file_suffix = get_file_suffix(dset_tag, group, num_train)
    label = f'tab:score{file_suffix}' if not pct_abstained else f'tab:pct_abstained{file_suffix}'
    filename = f"{'pct_abstained' if pct_abstained else 'score'}{file_suffix}_table.tex"
    make_table(len(column_names), rows, output_dir, caption=caption, label=label, filename=filename, header=header, use_siunitx=True)

def split_into_train_and_test(data, num_train):
    train_data, test_data = defaultdict(dict), defaultdict(dict)
    for dataset in data:
        for model in data[dataset]:
            labels, conf_levels, total_qs = data[dataset][model]
            # Shuffle labels and conf_levels together                                                                                           
            random.seed(2549900867) # All models should have the same split                                                                     
            combined = list(zip(labels, conf_levels))
            random.shuffle(combined)
            labels, conf_levels = map(np.array, zip(*combined))
            train_data[dataset][model] = (labels[:num_train], conf_levels[:num_train], num_train)
            test_data[dataset][model] = (labels[num_train:], conf_levels[num_train:], total_qs - num_train)
    return train_data, test_data
    
##############
# CALIBRATION
###############

def get_bins_bounds(conf_levels, n_bins=10, strategy='quantile'):
    if strategy == 'uniform':
        return np.linspace(0, 1, n_bins+1)
    else:
        return np.quantile(conf_levels, np.linspace(0, 1, n_bins+1))

def calibration_curve(labels, conf_levels, n_bins=10, strategy='quantile'):
    bin_bounds = get_bins_bounds(conf_levels, n_bins=n_bins, strategy=strategy)
    (bin_bounds[0], bin_bounds[-1]) = (0, 1) # sometimes np.quantile is a bit weird
    bin_lengths = np.array([bin_bounds[i+1] - bin_bounds[i] for i in range(len(bin_bounds)-1)])
    bin_correct = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    bin_conf_sum = np.zeros(n_bins)
    for i in range(len(labels)):
        for j in range(n_bins):
            if bin_bounds[j] <= conf_levels[i] < bin_bounds[j+1]:
                bin_total[j] += 1
                bin_correct[j] += labels[i]
                bin_conf_sum[j] += conf_levels[i]
                break
    # remove empty bins
    bin_lengths = bin_lengths[bin_total > 0]
    bin_pct_correct = bin_correct[bin_total > 0] / bin_total[bin_total > 0]
    bin_avg = bin_conf_sum[bin_total > 0] / bin_total[bin_total > 0]
    return bin_pct_correct, bin_avg, bin_lengths

def calibration_curve_plot(data, output_dir, group_tag, strategy='quantile', n_bins=10, num_train=None):
    plt.figure()
    for model in sort_models(data.keys()):
        if model not in style_per_model:
            style_per_model[model] = (linestyles.pop(0), colors.pop(0))
        (linestyle, color) = style_per_model[model]

        labels, conf_levels, _ = data[model]
        pct_correct, avg_msp, _ = calibration_curve(labels, conf_levels, n_bins=n_bins, strategy=strategy)
        if len(pct_correct) < n_bins:
            print("Model", model, f"has {n_bins-len(pct_correct)} empty bins, out of {n_bins} total bins.")
        plt.plot(avg_msp, pct_correct, label=f'{expand_model_name(model)}', linestyle=linestyle, linewidth=2, color=color)
    # Add black line on the diagonal to represent perfect calibration
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='-')
    make_and_sort_legend()
    file_suffix = get_file_suffix('', group_tag, num_train)
    finalize_plot(output_dir, 'msp', 'frac-correct', file_suffix= file_suffix)

def compute_calibration_error(labels, conf_levels, n_bins=10, strategy='quantile'):
    """Compute calibration error for a single dataset-model pair"""
    pct_correct, avg_msp, _ = calibration_curve(labels, conf_levels, n_bins=n_bins, strategy=strategy)
    return make_pct(np.mean(abs(pct_correct - avg_msp)))

def make_calibration_table(accs, calib_errors, model_names, output_dir, group, dset_tag, num_train=None):
    
    rows = []
    for i, model in enumerate(sort_models(model_names)):
        rows.append([expand_model_name(model), accs[i], calib_errors[i]])
    column_names = ['LLM', 'Q\&A accuracy', 'Calibration error']
    header = (' & '.join(column_names) + ' \\\\ \n'
              '\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \\cmidrule(lr){3-3}\n')
    caption = 'Calibration error and Q\&A accuracy for each model.'
    file_suffix = get_file_suffix(dset_tag, group, num_train)
    make_table(len(column_names), rows, output_dir, caption=caption, label=f'tab:calibration{file_suffix}', filename=f'calibration{file_suffix}_table.tex', header=header)

def calibration_scatter_plots(data, output_dir, group_tag, dset_tag, strategy='quantile', n_bins=10, num_train=None):
    ylabel = 'calib-platt' if 'platt' in group_tag else 'calib'
    avg_calib_errors, avg_accs, model_names = acc_scatter_plots(data, compute_calibration_error, output_dir, ylabel, group_tag, n_bins=n_bins, strategy=strategy)

    return avg_calib_errors

def platt_scaling_sigmoid(scores, A, B):
    """Apply Platt scaling: P(y=1|score) = 1 / (1 + exp(A * score + B))"""
    return expit(A * scores + B)

def fit_platt_scaling(labels, scores):
    """Fit Platt scaling parameters A and B using maximum likelihood"""
    labels = np.array(labels)
    scores = np.array(scores)
    P = labels.sum()
    N = len(labels) - P
    hi = (P + 1.0) / (P + 2.0)   # positive target
    lo = 1.0 / (N + 2.0)   # negative target
    targets = np.where(labels == 1, hi, lo) # regularised labels
    init_params = [0.0, 0.0]
    
    def neg_log_likelihood(params):
        A, B = params
        probs = platt_scaling_sigmoid(scores, A, B)
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.sum(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
    
    result = minimize(neg_log_likelihood, init_params, method='BFGS')
    
    return result.x[0], result.x[1]

def apply_platt_scaling(test_data, platt_params):
    """Apply Platt scaling to test data using parameters learned from train data"""
    platt_scaled_data = defaultdict(dict)
    
    for key in test_data:
        # We want this function to work with model-collapsed data (i.e. data[model] = stuff) or uncollapsed data (i.e. data[dataset][model] = stuff)
        if isinstance(test_data[key], dict):
            dataset = key
            for model in test_data[dataset]:
                test_labels, test_conf_levels, test_total_qs = test_data[dataset][model]
                (A, B) = platt_params[model]
                calibrated_conf_levels = platt_scaling_sigmoid(np.array(test_conf_levels), A, B)
                platt_scaled_data[dataset][model] = (test_labels, calibrated_conf_levels, test_total_qs)
        else:
            model = key
            test_labels, test_conf_levels, test_total_qs = test_data[model]
            (A,B) = platt_params[model]
            calibrated_conf_levels = platt_scaling_sigmoid(np.array(test_conf_levels), A, B)
            platt_scaled_data[model] = (test_labels, calibrated_conf_levels, test_total_qs)
    
    return platt_scaled_data
