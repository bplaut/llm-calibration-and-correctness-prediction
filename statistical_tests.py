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

def _collect_model_and_dataset_data(input_dir):
    # Collect all data indexed by data[group_name][dataset][model]
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], [], 0))))
    
    # Get all result files
    all_results = os.listdir(input_dir)
    
    for filename in all_results:
        # Match the new file pattern
        if re.match(r'd=.*_m=.*_q=.*_c=.*_p=.*\.txt', filename):
            filepath = os.path.join(input_dir, filename)
            
            # Parse filename to get components
            dataset, model, _ = parse_file_name(filename)
            
            # Extract conf_type and prompt_type from filename
            conf_type_match = re.search(r'_c=([^_]+)_', filename)
            prompt_type_match = re.search(r'_p=([^.]+)\.', filename)
            
            if conf_type_match and prompt_type_match:
                conf_type = conf_type_match.group(1)
                prompt_type = prompt_type_match.group(1)
                
                # Parse the data to get distributions
                labels, conf_levels_dists, total_qs = parse_data(filepath)
                
                if len(labels) > 0:
                    # Generate all confidence variants
                    variants = generate_all_confidence_variants(labels, conf_levels_dists, total_qs, conf_type, prompt_type)
                    
                    # Add each variant to all_data
                    for group_name, (variant_labels, variant_conf_levels, variant_total_qs) in variants.items():
                        old_labels, old_conf_levels, old_total_qs = all_data[group_name][dataset][model]
                        all_data[group_name][dataset][model] = (
                            np.concatenate([old_labels, variant_labels]), 
                            np.concatenate([old_conf_levels, variant_conf_levels]), 
                            old_total_qs + variant_total_qs
                        )
    
    return all_data

def conduct_mann_whitney_tests(input_dir, save_csv=False):
    all_data = _collect_model_and_dataset_data(input_dir)
    
    if not all_data:
        print("No data found. Please check the input directory and file format.")
        return
    
    # Organize results by dataset, model, and classifier type
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Optional: collect raw data for CSV
    if save_csv:
        test_data = {"group": [], "dataset": [], "model": [], "p_value": [], "u_stat": [], "reject": []}
    
    # Get all unique models
    all_models = set()
    for group_name in all_data:
        for dataset in all_data[group_name]:
            all_models.update(all_data[group_name][dataset].keys())
    
    if not all_models:
        print("No models found in the data.")
        return
    
    all_models = sort_models(list(all_models))
    for group_name in all_data:
        # Parse group name to identify classifier type
        conf_type, prompt_type, confidence_measure, _ = parse_group_name(group_name)
        
        classifier = conf_type.capitalize() + " " + confidence_measure.capitalize()
        
        for dataset in all_data[group_name]:
            for model in all_data[group_name][dataset]:
                labels, conf_levels, _ = all_data[group_name][dataset][model]
                if len(labels) > 0:
                    labels = np.array(labels)
                    conf_levels = np.array(conf_levels)
                    conf_levels_right = conf_levels[labels == 1]
                    conf_levels_wrong = conf_levels[labels == 0]
                    
                    if len(conf_levels_right) > 0 and len(conf_levels_wrong) > 0:
                        p_val, stat, verdict = mann_whitney(conf_levels_right, conf_levels_wrong)
                        results[classifier][dataset][model].append(p_val)
                        
                        if save_csv:
                            test_data["group"].append(group_name)
                            test_data["dataset"].append(dataset)
                            test_data["model"].append(model)
                            test_data["p_value"].append(p_val)
                            test_data["u_stat"].append(stat)
                            test_data["reject"].append(verdict)
    
    # Debug: print what groups were found
    print(f"\nFound {len(all_data)} unique metric groups")
    print(f"Found {len(all_models)} unique models")
        
    # Print summary tables
    print("\n" + "="*60)
    print("MANN-WHITNEY U TEST RESULTS")
    print("Number of tests with p < 10^{-4}")
    print("="*60)
    
    for classifier in results:
        # Individual dataset tables
        for dataset in sorted(results[classifier].keys()):
            print(f"\n{dataset}, {classifier}")
            print("-"*40)
            print(f"{'Model':<20} {'p < 1e-4':<15}")
            print("-"*40)
            dataset_name = format_dataset_name(dataset)
            
            for model in sort_models(results[classifier][dataset].keys()):
                p_vals = results[classifier][dataset][model]
                sig_count = sum(1 for p in p_vals if p < 10**(-4))
                total_count = len(p_vals)
                sig_str = f"{sig_count}/{total_count}" if total_count > 0 else "-"
                print(f"{model:<20} {sig_str:<15}")
    
        # Combined table across all datasets
        print(f"\n{'ALL DATASETS COMBINED'}, {classifier}")
        print("-"*40)
        print(f"{'Model':<20} {'p < 1e-4':<15}")
        print("-"*40)

        for model in all_models:
            p_vals = []
            for dataset in results[classifier]:
                if model in results[classifier][dataset]:
                    p_vals.extend(results[classifier][dataset][model])
            sig_count = sum(1 for p in p_vals if p < 1e-4)
            total_count = len(p_vals)
            sig_str = f"{sig_count}/{total_count}" if total_count > 0 else "-"
            print(f"{model:<20} {sig_str:<15}")
        
    # Optionally save raw data to CSV
    if save_csv:
        pd.DataFrame(test_data).to_csv("./stat_tests_output/mann_whitney.csv", index = False)
        print("\nRaw data saved to ./stat_tests_output/mann_whitney.csv")

def construct_confidence_intervals(input_dir):
    all_data = _collect_model_and_dataset_data(input_dir)
    
    test_data = {"group": [], "dataset": [], "model": [], "sample_auroc": [], "ci_lb": [], "ci_ub": []}
    
    for group_name in all_data:
        for dataset in all_data[group_name]:
            for model in all_data[group_name][dataset]:
                labels, conf_levels, _ = all_data[group_name][dataset][model]
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
                    test_data["group"].append(group_name)
                    test_data["dataset"].append(dataset)
                    test_data["model"].append(model)
                    test_data["sample_auroc"].append(sample_auroc)
                    test_data["ci_lb"].append(lower_bound)
                    test_data["ci_ub"].append(upper_bound)
                else:
                    print(f"Missing data for {group_name}, {dataset}, {model}")
    
    pd.DataFrame(test_data).to_csv("./stat_tests_output/confidence_intervals.csv", index = False)

def conduct_model_summary_tests(input_dir):
    all_data = _collect_model_and_dataset_data(input_dir)
    
    test_data = {"model": [], "group": [], "p_value": [], "t_stat": [], "t_dof": [], "w_stat": [], "reject": []}
    
    # Get all unique models across all groups and datasets
    all_models = set()
    for group_name in all_data:
        for dataset in all_data[group_name]:
            all_models.update(all_data[group_name][dataset].keys())
    
    for group_name in all_data:
        for model in all_models:
            model_aurocs = []
            for dataset in ALL_DATASETS:
                if dataset in all_data[group_name] and model in all_data[group_name][dataset]:
                    labels, conf_levels, _ = all_data[group_name][dataset][model]
                    if len(labels) > 0:
                        fpr, tpr, __ = roc_curve(labels, conf_levels)
                        auroc = auc(fpr, tpr)
                        model_aurocs.append(auroc)
                    else:
                        print(f"Missing data for {group_name}, {dataset}, {model}")
            
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
                test_data["group"].append(group_name)
    
    pd.DataFrame(test_data).to_csv("./stat_tests_output/summary_tests.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###
    # Options:
    # 0: Construct confidence interval on AUROC for every combination of model-dataset-group
    # 1: Mann-Whitney U test on every combination of model-dataset-group (prints summary tables)
    # 2: Summary t-tests/Wilcoxon for average AUROC across models
    # 3: Paired t-test for difference in scores (uses hardcoded data)
    ###
    parser.add_argument("--option", "-o", required = True, type = int, help = f"Integer from 0 to 2, determines which test to run")
    parser.add_argument("--input_dir", '-d', type=str, help="Input directory to read data from", required = True)
    args = parser.parse_args()

    if not os.path.exists("./stat_tests_output"):
        os.makedirs("./stat_tests_output")
    
    if args.option == 0:
        construct_confidence_intervals(args.input_dir) # Currently not doing these
    elif args.option == 1:
        conduct_mann_whitney_tests(args.input_dir, save_csv=True)
    elif args.option == 2:
        conduct_model_summary_tests(args.input_dir)
    else:
        raise ValueError("Invalid option. Please choose an integer from 0 to 2.")
