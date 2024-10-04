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

# We want to assign a style to each model globally, even when some models are missing from some groups (i.e., OpenAI models only have MSP, not Max Logit)
linestyles = ['-', ':', (0, (3, 1, 1, 1, 1, 1)), (0, (0.5,0.5,0.5,0.5,2)),(0, (5, 10)),(0, (5.5, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1)), (0, (0.25,0.25)), (0, (5,0.5,0.5,5)), (0, (0.5, 0.5)), (0,(1,1,1,3.5)), (0, (0.5,0.5,0.5,2)), (0,(2,1,2,2))]
colors = ['pink', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 'teal', 'slategray', 'indigo', '#bcbd22', 'black', '#17becf']
style_per_model = dict()

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
        plt.xlim([28, 89])
    if ylabel == 'auc':
        plt.ylim([50, 88])
    if ylabel in ('score', 'harsh-score'):
        plt.ylim([-15, 78])

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
    # If we're plotting size, exclude GPT models (because we don't know their size)
    if xlabel == 'size':
        xs = [xs[i] for i in range(len(xs)) if 'gpt' not in model_names[i].lower()]
        ys = [ys[i] for i in range(len(ys)) if 'gpt' not in model_names[i].lower()]
        model_names = [name for name in model_names if 'gpt' not in name.lower()]
    xs, ys = np.array(xs), np.array(ys)
    group = output_dir[output_dir.rfind('/')+1:]
    mark_color, marker, line_color = plot_style_for_group(group)
    scatter = plt.scatter(xs, ys, c=mark_color, marker=marker)
    texts = []

    for i in range(len(model_names)):
        texts.append(plt.text(xs[i], ys[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7))

    try:
        slope, intercept, r_value, p_value, std_err = linregress(xs, ys)
        print("slope, r_value, p_value for", xlabel, ylabel, "is", slope, r_value, p_value)
        plt.plot(xs, intercept + slope * xs, color=line_color, linestyle='-')

        plot_name = 'MSP' if group == 'no_abst_norm_logits' else 'Max Logit' if group == 'no_abst_raw_logits' else '' if group == 'main_figs' else group
        plot_name = plot_name if dataset == 'all datasets' else f'{plot_name}, {dataset}'
        file_suffix = f"_{dataset}_{plot_name.replace(' ','_').replace(',','')}"
        if file_suffix.endswith('_'):
            file_suffix = file_suffix[:-1]
        finalize_plot(output_dir, xlabel, ylabel, title_suffix=f': {plot_name} (r,p = {r_value:.3f},{p_value:.5f})', file_suffix=file_suffix, texts=texts)
    except ValueError as e:
        print("ValueError when running linear regression on", xlabel, ylabel, ":", e)

           
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
    # define twelve unique styles

    result_thresholds, result_scores, base_scores, pcts_abstained = dict(), dict(), dict(), dict()
    for model in sort_models(data.keys()):
        (xs, ys, pct_abstained_all_threshes) = data[model]
        if model not in style_per_model:
            style_per_model[model] = (linestyles.pop(0), colors.pop(0))
        (linestyle, color) = style_per_model[model]
        # Mark the provided threshold if given, else mark the threshold with the best score
        if model in thresholds_to_mark:
            thresh_to_mark = thresholds_to_mark[model]
            thresh_idx = np.where(xs == thresh_to_mark)[0][0] # First zero idx is because np.where returns a tuple, second zero idx is because we only want the first index (although there should only be one)
        else:
            thresh_idx = np.argmax(ys)
            thresh_to_mark = xs[thresh_idx]
        score_to_mark = ys[thresh_idx]
        pct_abstained = pct_abstained_all_threshes[thresh_idx]
        # zorder determines which objects are on top
        plt.scatter([thresh_to_mark], [score_to_mark], color='black', marker='o', s=20, zorder=3)
        base_score = ys[0] # We added -1 to the front for base score, see plot_score_vs_thresholds
        plt.plot(xs, ys, label=f"{expand_model_name(model)}", zorder=2, linestyle=linestyle, linewidth=2, color=color)
        result_thresholds[model] = thresh_to_mark
        base_scores[model] = base_score
        result_scores[model] = score_to_mark
        pcts_abstained[model] = pct_abstained

    if 'norm' in output_dir and ylabel == 'harsh-score':
        # Only make legend for bottom left plot in paper
        make_and_sort_legend()
        plt.legend(handlelength=2.5)
    plot_name = 'MSP' if 'norm' in output_dir else 'Max Logit' if 'raw' in output_dir else 'unknown'
    plot_name = plot_name if dataset == 'all datasets' else f'{plot_name}, {dataset}'
    finalize_plot(output_dir, xlabel, ylabel, title_suffix = f': {plot_name}', file_suffix = f'_{dataset}')
    return result_thresholds, result_scores, base_scores, pcts_abstained
    
def plot_score_vs_thresholds(data, output_dir, datasets, wrong_penalty=1, thresholds_to_mark=dict()):
    # Inner max is for one model + dataset, middle max is for one dataset, outer max is overall
    max_conf = max([max([max(conf_levels) for _, (_,conf_levels,_) in data[dataset].items()])
                    for dataset in datasets])
    thresholds = np.linspace(0, max_conf, 200) # 200 data points per plot
    thresholds = np.append([-0.0001], thresholds) # The base score is the score when the threshold is 0, but this could cause float issues if confidence is also exactly zero at times. So add another data point

    if abs(max_conf - 1) < 0.01: # We're dealing with probabilities: add more points near 1
        thresholds = np.append(thresholds, np.linspace(0.99, 1, 100))
    # Add all keys in thresholds_to_mark to thresholds, and sort
    thresholds = np.sort(np.unique(np.append(thresholds, list(thresholds_to_mark.values()))))

    # For each model and dataset, compute the score and pct abstained for each threshold
    all_scores = defaultdict(lambda: defaultdict(list))
    all_pcts_abstained = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        for model, (labels, conf_levels, total_qs) in data[dataset].items():
            scores  = []
            pcts_abstained = []
            for thresh in thresholds:
                scores.append(compute_score(labels, conf_levels, total_qs, thresh, wrong_penalty))
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
            
    dataset_name = 'all datasets' if len(datasets) > 1 else datasets[0]
    ylabel = 'score' if wrong_penalty == 1 else 'harsh-score' if wrong_penalty == 2 else f'Wrong penalty of {wrong_penalty}'
    return score_plot(overall_results, output_dir, 'conf', ylabel, dataset_name, thresholds_to_mark)

def train_and_test_score_plots(test_data, train_data, output_dir, datasets, wrong_penalty=1):
    # Get optimal thresholds for train data, use those to compute scores for test data
    (optimal_train_thresholds, _, _, _) = plot_score_vs_thresholds(train_data, os.path.join(output_dir, 'train'), datasets, wrong_penalty=wrong_penalty)
    (_, test_scores, base_test_scores, pcts_abstained) = plot_score_vs_thresholds(test_data, os.path.join(output_dir, 'test'), datasets, wrong_penalty=wrong_penalty, thresholds_to_mark=optimal_train_thresholds)
    return optimal_train_thresholds, test_scores, base_test_scores, pcts_abstained

def make_auroc_table(msp_group_data, max_logit_group_data, output_dir, dataset=''):
    model_results_msp = make_model_dict(*msp_group_data)
    model_results_max_logit = make_model_dict(*max_logit_group_data)
    rows = []
    # Sort the rows by model series, then by model size. Also put gpt models at the end
    for model in sort_models(model_results_msp.keys()):
        # Default is (0, '--', {}) if we don't have results for that model. This can happen with e.g. GPT models where we only have MSP results, not raw logits. The '--' will actually end up in the table in those cases
        (auc_msp, acc_msp, _) = model_results_msp.get(model, ('--', 0, {}))
        (auc_max_logit, acc_max_logit, _) = model_results_max_logit.get(model, ('--', 0, {}))
        if abs(acc_msp - acc_max_logit) > 0.01 and 'gpt' not in model:
            print(f"Warning: accuracies for {model} don't match: {acc_msp} vs {acc_max_logit}")
        rows.append([expand_model_name(model), acc_msp, auc_msp, '2/2', auc_max_logit, '2/2'])
    column_names = ['LLM', 'Q\\&A Accuracy', 'AUROC', '$p < 10^{-4}$', 'AUROC', '$p < 10^{-4}$']
    header = ('& & \\multicolumn{2}{c}{MSP} & \\multicolumn{2}{c}{Max Logit} \\\\ \n'
              + ' & '.join(column_names) + ' \\\\ \n'
              + '\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \n')
    caption = 'AUROC results for %s. See Table~\\ref{tab:arc_auroc} for more explanation.' % format_dataset_name(dataset)
    dataset_for_label = '' if dataset == '' else f'{dataset}_'
    make_table(len(column_names), rows, output_dir, caption=caption, label=f'tab:{dataset_for_label}auroc', filename=f'{dataset_for_label}auroc_table.tex', header=header)

def make_score_table(msp_group_data, max_logit_group_data, output_dir, dataset='', pct_abstained=False):
    # If pct_abstained=True, we'll write the pct_abstained instead of the score
    model_results_msp = make_model_dict(*msp_group_data)
    model_results_ml = make_model_dict(*max_logit_group_data)
    rows = []
    for model in sort_models(model_results_msp.keys()):
        # Default is (0, 0, {}) if we don't have results for that model
        (_, _, score_data_msp) = model_results_msp.get(model, (0, 0, {}))
        (_, _, score_data_ml) = model_results_ml.get(model, (0, 0, {}))
        rows.append([expand_model_name(model)])
        for wrong_penalty in score_data_msp:
            # The '--' will actually end up in the table in some cases
            (_, score_msp, base_score_msp, pct_abstained_msp) = score_data_msp.get(wrong_penalty, (0, '--', 0, '--'))
            (_, score_ml, base_score_ml, pct_abstained_ml) = score_data_ml.get(wrong_penalty, (0, '--', 0, '--'))
            if abs(base_score_msp - base_score_ml) > 0.01 and 'gpt' not in model:
                print(f"Warning: base scores for {model} don't match: {base_score_msp} vs {base_score_ml}")
            if pct_abstained:
                rows[-1].extend([0, pct_abstained_msp, pct_abstained_ml])
            else:
                rows[-1].extend([base_score_msp, score_msp, score_ml])
    column_names = ['LLM', 'Base', 'MSP', 'Max Logit', 'Base', 'MSP', 'Max Logit']
    header = ('& \\multicolumn{3}{c}{Balanced} & \\multicolumn{3}{c}{Conservative} \\\\ \n'
              + ' & '.join(column_names) + ' \\\\ \n'
              + '\\cmidrule(lr){1-1}\\cmidrule(lr){2-4}\\cmidrule(lr){5-7} \n')
    caption = ('Q\\&A with abstention results for %s. See Table~\\ref{tab:score} for an explanation of the scoring scheme.' if not pct_abstained else 'Frequency of abstention on %s in the Section~\\ref{sec:abstain} experiments.') % format_dataset_name(dataset)
    label = f'tab:{dataset}_score' if not pct_abstained else f'tab:{dataset}_pct_abstained'
    dataset_for_label = '' if dataset == '' else f'{dataset}_'
    filename = dataset_for_label + ('pct_abstained' if pct_abstained else 'score') + '_table.tex'
    make_table(len(column_names), rows, output_dir, caption=caption, label=label, filename=filename, header=header)

def get_bins_bounds(conf_levels, n_bins=10, strategy='uniform'):
    if strategy == 'uniform':
        return np.linspace(0, 1, n_bins+1)
    else:
        return np.quantile(conf_levels, np.linspace(0, 1, n_bins+1))

def calibration_curve(labels, conf_levels, n_bins=10, strategy='uniform'):
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

def calibration_plot(data, output_dir, strategy='quantile', n_bins=10):
    # strategy is either 'quantile' or 'uniform'
    plt.figure()
    for model in sort_models(data.keys()):
        if model not in style_per_model:
            style_per_model[model] = (linestyles.pop(0), colors.pop(0))
        (linestyle, color) = style_per_model[model]

        labels, conf_levels, _ = data[model]
        pct_correct, avg_msp, _ = calibration_curve(labels, conf_levels, n_bins=n_bins, strategy=strategy)
        # absolute_error = f'{np.mean(abs(pct_correct - avg_msp)):.2f}'
        if len(pct_correct) < n_bins:
            print("Model", model, f"has {n_bins-len(pct_correct)} empty bins, out of {n_bins} total bins.")
        plt.plot(avg_msp, pct_correct, label=f'{expand_model_name(model)}', linestyle=linestyle, linewidth=2, color=color)
    # Add black line on the diagonal to represent perfect calibration
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='-')
    make_and_sort_legend()
    plt.legend()
    finalize_plot(output_dir, 'msp', 'frac-correct', title_suffix=': Calibration', file_suffix=f'_{strategy}')

def make_calibration_table(data, output_dir, strategy='quantile', n_bins=10):
    rows = []
    for model in sort_models(data.keys()):
        labels, conf_levels, _ = data[model]
        pct_correct, avg_msp, bin_lengths = calibration_curve(labels, conf_levels, n_bins=n_bins, strategy=strategy)
        absolute_error = np.mean(abs(pct_correct - avg_msp))
        # sum of bin lengths might not be 1 if some are empty
        rows.append([expand_model_name(model), absolute_error])
    column_names = ['LLM', 'Expected calibration error']
    header = (' & '.join(column_names) + ' \\\\ \n'
              '\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \n')
    caption = 'Calibration Error for each model. See Table~\\ref{tab:calibration} for more explanation.'
    make_table(len(column_names), rows, output_dir, caption=caption, label='tab:calibration', filename=f'calibration_table_{strategy}.tex', header=header, precision=2)

def calibration_acc_plot(data, output_dir, strategy='uniform', n_bins=10):
    model_data = defaultdict(lambda: [])
    for group in data:
        if 'norm' in group:
            for dataset in data[group]:
                for model in data[group][dataset]:
                    labels, conf_levels, _ = data[group][dataset][model]
                    pct_correct, avg_msp, _ = calibration_curve(labels, conf_levels, n_bins=n_bins, strategy=strategy)
                    absolute_error = np.mean(abs(pct_correct - avg_msp))
                    accuracy = make_pct(np.mean(labels))
                    model_data[model].append((accuracy, absolute_error))

    avg_accs, avg_calib_errors, model_names = [], [], []
    for model in sort_models(model_data.keys()):
        (accs, calib_errors) = zip(*model_data[model])
        avg_acc = np.mean(accs)
        avg_calib_error = np.mean(calib_errors)
        avg_accs.append(avg_acc)
        avg_calib_errors.append(avg_calib_error)
        model_names.append(model)

    dataset_name = 'all datasets'
    model_sizes = [model_size(model) for model in model_names]
    scatter_plot(avg_accs, avg_calib_errors, output_dir, model_names, 'acc', 'calib', dataset_name)
    scatter_plot(model_sizes, avg_calib_errors, output_dir, model_names, 'size', 'calib', dataset_name)
    
def make_dataset_plots(all_data, output_dir):
    # one entry per dataset, containing avg acc, avg MSP auc, and avg max logit auc
    # all_data[group][dataset][model] = (labels, conf_levels, total_qs)
    dataset_stats = defaultdict(lambda: ([], [], []))
    # First collect all the data for each dataset
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
    # Then average the data for each dataset
    for dataset in dataset_stats:
        accs, msp_aucs, max_logit_aucs = dataset_stats[dataset]
        dataset_stats[dataset] = (np.mean(accs), np.mean(msp_aucs), np.mean(max_logit_aucs))
        
    # Make table
    rows = []
    for dataset in sorted(dataset_stats.keys()):
        avg_acc, msp_auc, max_logit_auc = dataset_stats[dataset]
        rows.append([format_dataset_name(dataset), avg_acc, msp_auc, max_logit_auc])
    column_names = ['Dataset', 'Q\\&A Accuracy', 'MSP AUROC', 'Max Logit AUROC']
    prefix = ('no' if 'no_abst' in list(all_data.keys())[0] else 'yes') + '_abst'
    filename = prefix + '_dataset.tex'
    header = (' & ' + ' & '.join(column_names[1:]) + ' \\\\ \n'
              '\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \\cmidrule(lr){3-3} \\cmidrule(lr){4-4}\n')
    make_table(len(column_names), rows, output_dir, caption='Average Q\\&A accuracy and AUROCs per dataset. All values are percentages, averaged over the then models and two prompts.', label='tab:dataset', filename=filename, header=header)

    # Make bar graph. Three segments on the x-axis: Q&A accuracy, MSP AUROC, Max Logit AUROC. Within each segment, one bar per dataset. So there should be three segments, each with 5 bars
    labels = ['Q&A Accuracy', 'MSP AUROC', 'Max Logit AUROC']
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
    filepath = os.path.join(output_dir, f'{prefix}_dataset_bar.pdf')
    plt.savefig(filepath)
    print("Dataset bar graph saved -->", filepath)
    plt.close()
    
def make_table(num_cols, rows, output_dir, caption='', label='', filename='table.tex', header='', precision=1):
    filename = os.path.join(output_dir, filename)
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, 'w') as f:
        f.write('\\begin{table*}[h]\n')
        f.write('\\centering\n')
        f.write(f'\\caption{{{caption}}}\n')
        f.write(f'\\label{{{label}}}\n')
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
            # total_qs can differ from len(labels) == len(conf_levels) if we allowed the base LLM to abstain, because total_qs counts abstentions but len(labels) doesn't. But we're just using no-abstention data, so we can ignore this.
            num_train = 20
            train_data[dataset][model] = (labels[:num_train], conf_levels[:num_train], num_train)
            test_data[dataset][model] = (labels[num_train:], conf_levels[num_train:], total_qs - num_train)
            
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

    aucs, accs, model_names = auc_acc_plots(data, all_aucs, output_dir)
    return score_data, aucs, accs, model_names

def collapse_data_to_model(data, logit_type='norm_logits'):
    # Input: dict of the form data[group][dataset][model] = (labels, conf_levels, total_qs)
    # Output: dict of the form data[model] = (labels, conf_levels, total_qs)
    collapsed = defaultdict(lambda: ([], [], 0))
    for group in data:
        if logit_type in group:
            for dataset in data[group]:
                for model in data[group][dataset]:
                    (labels, conf_levels, total_qs) = data[group][dataset][model]
                    old_labels, old_conf_levels, old_total_qs = collapsed[model]
                    collapsed[model] = (np.concatenate([old_labels, labels]), np.concatenate([old_conf_levels, conf_levels]), old_total_qs + total_qs)
    return collapsed

def merge_groups(group_data):
    # Merge to a single "group" based on the means across groups
    all_auc_acc_data = defaultdict(lambda: ([], []))
    all_score_data = defaultdict(lambda: (defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)))
    for group in group_data:
        (score_data, aucs, accs, model_names) = group_data[group]
        for i, model_name in enumerate(model_names):
            # Collect the auc and acc for this model in each group into a list
            all_auc_acc_data[model_name][0].append(aucs[i])
            all_auc_acc_data[model_name][1].append(accs[i])
            
        for wrong_penalty in score_data:
            # Same idea here, except these are each dicts with model name as the key
            (thresholds, our_scores, base_scores, pcts_abstained) = score_data[wrong_penalty]
            for model in thresholds:
                all_score_data[wrong_penalty][0][model].append(thresholds[model])
                all_score_data[wrong_penalty][1][model].append(our_scores[model])
                all_score_data[wrong_penalty][2][model].append(base_scores[model])
                all_score_data[wrong_penalty][3][model].append(pcts_abstained[model])

    avg_aucs, avg_accs, model_names = [], [], []
    for model_name, (aucs, accs) in all_auc_acc_data.items():
        avg_aucs.append(np.mean(aucs))
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
    return new_score_data, avg_aucs, avg_accs, model_names

def make_model_dict(score_data, aucs, accs, model_names):
    # Change the dict structure so that the model is the key
    model_results = dict()
    for i, model_name in enumerate(model_names):
        model_score_data = dict()
        for wrong_penalty in score_data:
            thresholds, our_scores, base_scores, pcts_abstained = score_data[wrong_penalty]
            thresh, our_score, base_score, pct_abstained = thresholds[model_name], our_scores[model_name], base_scores[model_name], pcts_abstained[model_name]
            model_score_data[wrong_penalty] = (thresh, our_score, base_score, pct_abstained)
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
            texts.append(plt.text(accs[i], aucs[i], expand_model_name(model_names[i]), ha='right', va='bottom', alpha=0.7, fontsize='small'))

    file_suffix = '-' + '-'.join(group_data.keys())
    if 'prompt' in file_suffix:
        logit_type, _ = group_label(list(group_data.keys())[0])
        title_suffix = ': ' + logit_type + ', prompt comparison'
    else:
        title_suffix = ': ' + ' and '.join([group_label(group)[0] for group in group_data])
    plt.legend()
    finalize_plot(output_dir, 'acc', 'auc', file_suffix=file_suffix, title_suffix=title_suffix, texts=texts)
    
    # Second plot: AUC vs accuracy, averaged across groups
    score_data, avg_aucs, avg_accs, model_names = merge_groups(group_data)
    model_sizes = [model_size(model) for model in model_names]
    scatter_plot(avg_accs, avg_aucs, output_dir, model_names, 'acc', 'auc')
    scatter_plot(model_sizes, avg_aucs, output_dir, model_names, 'size', 'auc')
    scatter_plot(model_sizes, avg_accs, output_dir, model_names, 'size', 'acc')
    
def main():
    # Setup
    if len(sys.argv) < 5:
        print("Usage: python plot_data.py <output_directory> <dataset1,dataset2,...> <collapse_prompts> <data_file1> [<data_file2> ...]")
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
    print(f"Reading from {len(file_paths)} files...")
    incl_unparseable = True # We've decided to always include unparseable questions, but leaving this here in case we want to change it in the future for some reason


    # Data aggregation. We want all_data[group][dataset][model] = (labels, conf_levels, total_qs)
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], 0))))
    for file_path in file_paths:
        dataset, model, group = parse_file_name(os.path.basename(file_path), collapse_prompts)
        if dataset in datasets_to_analyze:
            labels, conf_levels, total_qs = parse_data(file_path, incl_unparseable)
            old_labels, old_conf_levels, old_total_qs = all_data[group][dataset][model]
            all_data[group][dataset][model] = (np.concatenate([old_labels, labels]), np.concatenate([old_conf_levels, conf_levels]), old_total_qs + total_qs)
            
    # Non-group based plots
    make_dataset_plots(all_data, output_dir)
    calibration_plot(collapse_data_to_model(all_data), output_dir)
    make_calibration_table(collapse_data_to_model(all_data), output_dir)
    calibration_acc_plot(all_data, output_dir)

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
                        make_score_table(msp_group, max_logit_group, new_output_dir, dataset=dset, pct_abstained=True)
                        
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
        make_score_table(new_group1, new_group2, new_output_dir, dset, pct_abstained=True)
    except KeyError:
        print("\nCouldn't find the right groups for the overall average plot, skipping.\n")

if __name__ == "__main__":
    main()
