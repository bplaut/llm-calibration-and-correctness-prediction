import scipy.stats as stats
from utils import *
import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from collections import defaultdict


ALL_DATASETS = ["arc", "hellaswag", "mmlu", "truthfulqa", "winogrande"]
ALL_MODELS = ["Falcon-7b", "Falcon-40b", "gpt-3.5-turbo", "gpt-4-turbo", "Llama-7b", "Llama-13b", "Llama-70b", "Llama3-8b", "Llama3-70b", "Mistral", "Mixtral", "Solar", "Yi-6b", "Yi-34b"]
ALL_PROMPTS = ["first_prompt", "second_prompt"]
ALL_VALUES = ["raw_logits", "norm_logits"]

def test_large_sample(data, threshold = 30):
    return len(data), len(data) >= threshold

def test_normality(data, mode = "ks", threshold = 0.05):
    """
    Shapiro-Wilk: smaller datasets
    Kolmogorov-Smirnov: bigger datasets
    """
    if mode == "sw":
        p_value = stats.shapiro(data).pvalue
    elif mode == "ks":
        p_value = stats.kstest(data, "norm").pvalue
    return p_value, p_value > threshold

def test_equal_variance(data1, data2, mode = "bartlett", threshold = 0.05):
    """
    Levene: less sensitive
    Bartlett: more sensitive
    """
    if mode == "lv":
        p_value = stats.levene(data1, data2).pvalue
    elif mode == "bl":
        p_value = stats.bartlett(data1, data2).pvalue
    return p_value, p_value > threshold

def mann_whitney(data1, data2, threshold = 0.05):
    u_statistic, u_p_value = stats.mannwhitneyu(data1, data2)
    return u_p_value, u_statistic, u_p_value < threshold

def unpaired_z(data1, data2, threshold = 0.05):
    z_statistic, z_p_value = stats.ttest_ind(data1, data2, equal_var = True)
    return z_p_value, z_statistic, z_p_value < threshold

def one_sample_t(data, expected_mean, alternative = "two-sided", threshold = 0.05):
    test_result = stats.ttest_1samp(data, expected_mean, alternative = alternative)
    return test_result.pvalue, test_result.statistic, test_result.df, test_result.pvalue < threshold

def wilcoxon(data, expected_mean, alternative = "two-sided", threshold = 0.05):
    diffs = list(map(lambda x: x - expected_mean, data))
    test_result = stats.wilcoxon(diffs, alternative = alternative)
    return test_result.pvalue, test_result.statistic, test_result.pvalue < threshold

def test_assumptions(data1, data2):
    _, sample1_large = test_large_sample(data1)
    _, sample2_large = test_large_sample(data2)
    large_sample = sample1_large and sample2_large
    _, normal1 = test_normality(data1)
    _, normal2 = test_normality(data2)
    normal = normal1 and normal2
    equal_variance = test_equal_variance(data1, data2)
    return (large_sample or normal) and equal_variance

def build_confidence_interval(data, alpha = 0.05):
    lower_percentile = alpha * 50
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return lower_bound, upper_bound

def _collect_model_and_dataset_data(incl_unparseable, input_dir):
    # index data by data[prompt][value][dataset][model]
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], 0)))))
    all_results = os.listdir(input_dir)
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    results_pattern = fr"{dataset}_{model}-.*_no_abst_{value}_{prompt}.txt"
                    relevant_files = [filename for filename in all_results if re.match(results_pattern, filename)]
                    for filename in relevant_files:
                        labels, conf_levels, total_qs = parse_data(os.path.join(input_dir, filename), incl_unparseable)
                        # print("Labels")
                        # print(labels)
                        # print("Conf levels")
                        # print(conf_levels)
                        # print("total qs")
                        # print(total_qs)
                        old_labels, old_conf_levels, old_total_qs = all_data[prompt][value][dataset][model]
                        # print("Old labels")
                        # print(old_labels)
                        # print("Old conf levels")
                        # print(old_conf_levels)
                        # print("Old total qs")
                        # print(old_total_qs)
                        all_data[prompt][value][dataset][model] = (np.concatenate([old_labels, labels]), np.concatenate([old_conf_levels, conf_levels]), old_total_qs + total_qs)
    return all_data

def conduct_mann_whitney_tests(incl_unparseable, input_dir):
    all_data = _collect_model_and_dataset_data(incl_unparseable, input_dir)
    # for prompt in ALL_PROMPTS:
    #     for value in ALL_VALUES:
    #         for dataset in ALL_DATASETS:
    #             for model in ALL_MODELS:
    #                 try:
    #                     print(len(all_data[prompt][value][dataset][model][0]))
    #                 except KeyError:
    #                     print(prompt, value, dataset, model)
    test_data = {"prompt": [], "value": [], "dataset": [], "model": [], "p_value": [], "u_stat": [], "reject": []}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    labels, conf_levels, _ = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        labels = np.array(labels)
                        conf_levels = np.array(conf_levels)
                        conf_levels_right = conf_levels[labels == 1]
                        conf_levels_wrong = conf_levels[labels == 0]
                        p_val, stat, verdict = mann_whitney(conf_levels_right, conf_levels_wrong)
                        test_data["prompt"].append(prompt)
                        test_data["value"].append(value)
                        test_data["dataset"].append(dataset)
                        test_data["model"].append(model)
                        test_data["p_value"].append(p_val)
                        test_data["u_stat"].append(stat)
                        test_data["reject"].append(verdict)
                    else:
                        if "gpt" not in model or value != "raw_logits":
                            print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
    pd.DataFrame(test_data).to_csv("./stat_tests_output/mann_whitney.csv", index = False)

def construct_confidence_intervals(incl_unparseable, input_dir):
    all_data = _collect_model_and_dataset_data(incl_unparseable, input_dir)
    test_data = {"prompt": [], "value": [], "dataset": [], "model": [], "sample_auroc": [], "ci_lb": [], "ci_ub": []}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    labels, conf_levels, _ = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        fpr, tpr, __ = roc_curve(labels, conf_levels)
                        sample_auroc = auc(fpr, tpr)
                        bootstrapped_aurocs = []
                        labels = np.array(labels)
                        conf_levels = np.array(conf_levels)
                        for i in range(1000):
                            indices = resample(np.arange(len(labels)))
                            bootstrapped_labels = labels[indices]
                            bootstrapped_conf_levels = conf_levels[indices]
                            fpr, tpr, ___ = roc_curve(bootstrapped_labels, bootstrapped_conf_levels)
                            bootstrapped_aurocs.append(auc(fpr, tpr))
                        lower_bound, upper_bound = build_confidence_interval(bootstrapped_aurocs)
                        test_data["prompt"].append(prompt)
                        test_data["value"].append(value)
                        test_data["dataset"].append(dataset)
                        test_data["model"].append(model)
                        test_data["sample_auroc"].append(sample_auroc)
                        test_data["ci_lb"].append(lower_bound)
                        test_data["ci_ub"].append(upper_bound)
                    else:
                        print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
    pd.DataFrame(test_data).to_csv("./stat_tests_output/confidence_intervals.csv", index = False)

def conduct_model_summary_tests(incl_unparseable, input_dir):
    all_data = _collect_model_and_dataset_data(incl_unparseable, input_dir)
    test_data = {"model": [], "prompt": [], "value": [], "p_value": [], "t_stat": [], "t_dof": [], "w_stat": [], "reject": []}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for model in ALL_MODELS:
                model_aurocs = []
                for dataset in ALL_DATASETS:
                    labels, conf_levels, _ = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        fpr, tpr, __ = roc_curve(labels, conf_levels)
                        auroc = auc(fpr, tpr)
                        model_aurocs.append(auroc)
                    else:
                        print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
                if len(model_aurocs) > 0:
                    _, is_normal = test_normality(model_aurocs)
                    if is_normal:
                        p_val, stat, df, verdict = one_sample_t(model_aurocs, 0.5, alternative = "greater")
                        test_data["p_value"].append(p_val)
                        test_data["t_stat"].append(stat)
                        test_data["t_dof"].append(df)
                        test_data["w_stat"].append(np.nan)
                        test_data["reject"].append(verdict)
                    else:
                        p_val, stat, verdict = wilcoxon(model_aurocs, 0.5, alternative = "greater")
                        test_data["p_value"].append(p_val)
                        test_data["t_stat"].append(np.nan)
                        test_data["t_dof"].append(np.nan)
                        test_data["w_stat"].append(stat)
                        test_data["reject"].append(verdict)
                    test_data["model"].append(model)
                    test_data["prompt"].append(prompt)
                    test_data["value"].append(value)
    pd.DataFrame(test_data).to_csv("./stat_tests_output/summary_tests.csv", index = False)

def collate_paired_t_test_data(all_data):
    test_data = {"prompt": [], "value": [], "dataset": [], "model": [], "num_questions": [], "num_base_correct": [], "num_base_wrong": []}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    labels, conf_levels, total_qs = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        test_data["prompt"].append(prompt)
                        test_data["value"].append(value)
                        test_data["dataset"].append(dataset)
                        test_data["model"].append(model)
                        test_data["num_questions"].append(total_qs)
                        test_data["num_base_correct"].append(np.count_nonzero(labels == 1))
                        test_data["num_base_wrong"].append(np.count_nonzero(labels == 0))
                    else:
                        print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
    return test_data

def conduct_paired_t_tests(incl_unparseable, input_dir):
    # wait recompute thresholds? and how to split up?
    # all_data = _collect_model_and_dataset_data(incl_unparseable, input_dir)
    # t_test_data = collate_paired_t_test_data(all_data)
    score_types = ["base", "msp", "max_logit"]
    balanced = {st: [] for st in score_types}
    conservative = {st: [] for st in score_types}
    sections = {"balanced": balanced, "conservative": conservative}
    balanced["base"] = [-40.4, -9.7, -16.9, -7.9, 18.2, 16.9, 37.6, 34.2, 2.6, 36.2]
    balanced["msp"] = [0, 1.9, -0.2, 6.6, 24.1, 19.6, 38.1, 34.4, 12.2, 38.0]
    balanced["max_logit"] = [0, 0.7, 0.1, 6.6, 21.4, 18.7, 38.3, 34.5, 10.2, 36.6]
    conservative["base"] = [-108.8, -63.2, -75.1, -61.8, -22.3, -21.4, 9.5, 3.4, -43, 5.2]
    conservative["msp"] = [0, -0.5, 0, 1.2, 9.2, 0.6, 17.3, 6.4, 4, 18.6]
    conservative["max_logit"] = [0, 0, 0, 0.2, 4.9, 5.6, 17.2, 12.4, 1.8, 15.6]
    for s in sections:
        print(s)
        for st in score_types[1:]:
            _, p_val = stats.ttest_rel(sections[s]["base"], sections[s][st])
            print("Comparing based with", st, round(p_val, 5))
        print()

def conduct_auroc_t_test(incl_unparseable, input_dir):
    # testing if MSP aurocs > max logit aurocs
    all_data = _collect_model_and_dataset_data(incl_unparseable, input_dir)
    value_aurocs = {v: [] for v in ALL_VALUES}
    for prompt in ALL_PROMPTS:
        for value in ALL_VALUES:
            for dataset in ALL_DATASETS:
                for model in ALL_MODELS:
                    labels, conf_levels, _ = all_data[prompt][value][dataset][model]
                    if len(labels) > 0:
                        fpr, tpr, __ = roc_curve(labels, conf_levels)
                        auroc = auc(fpr, tpr)
                        value_aurocs[value].append(auroc)
                    else:
                        print(f"Missing data for {prompt}, {dataset}, {model}, {value}")
    _, p_value = stats.ttest_ind(value_aurocs["norm_logits"], value_aurocs["raw_logits"], alternative = "greater")
    print("Testing if MSP AUROCs > max logit AUROCs")
    print("P-value:", round(p_value, 4))
    print("MSP AUROC avg:", round(np.mean(value_aurocs["norm_logits"]), 3))
    print("Max logit AUROC avg:", round(np.mean(value_aurocs["raw_logits"]), 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###
    # Options:
    # 0: Construct confidence interval on AUROC for every combination of model-dataset-prompt (200)
    # 1: Mann-Whitney U test on every combination of model-dataset-prompt (200)
    # 2: Summary t-tests/Wilcoxon for average AUROC across models (40)
    # 3: Paired t-test for difference in scores for model-dataset-prompt-value (200)
    # 4: MSP vs Max Logit aurocs battle royale
    ###
    parser.add_argument("--option", "-o", required = True, type = int, help = f"Integer from 0 to 4, determines which test to run")
    parser.add_argument("--input_dir", '-d', type=str, help="Input directory to read data from", required = True)
    args = parser.parse_args()
    incl_unparseable = True # We've decided to always include unparseable questions, but leaving this here in case we want to change it in the future for some reason

    if not os.path.exists("./stat_tests_output"):
        os.makedirs("./stat_tests_output")
    
    if args.option == 0:
        construct_confidence_intervals(incl_unparseable, args.input_dir)
    elif args.option == 1:
        conduct_mann_whitney_tests(incl_unparseable, args.input_dir)
    elif args.option == 2:
        conduct_model_summary_tests(incl_unparseable, args.input_dir)
    elif args.option == 3:
        conduct_paired_t_tests(incl_unparseable, args.input_dir)
    elif args.option == 4:
        conduct_auroc_t_test(incl_unparseable, args.input_dir)
