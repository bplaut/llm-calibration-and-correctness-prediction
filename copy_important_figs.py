import os
import shutil
import sys

def copy_files(output_directory, filepaths):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filepath in filepaths:
        extensions = ['.pdf', '.png', '.tex']
        for extension in extensions:
            full_path = filepath + extension
            if os.path.isfile(full_path):
                # Copy file to output directory
                base_name = os.path.basename(full_path)[:-4]
                dir_name = os.path.dirname(full_path)
                prompt_str = '' if 'prompt' in base_name else '_first_prompt' if 'first_prompt' in dir_name else '_second_prompt' if 'second_prompt' in dir_name else ''
                logit_str = '' if ('logit' in base_name.lower() or 'MSP' in base_name) else '_norm_logits' if 'norm_logits' in dir_name else '_raw_logits' if 'raw_logits' in dir_name else ''
                new_path = os.path.join(output_directory, base_name + prompt_str + logit_str + extension)
                shutil.copy(full_path, new_path)
                print("Successfully copied", new_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python copy_important_figs.py <input_dir> <output_directory>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    cross_group_dir = input_dir + '/main_figs/cross_group_plots'
    suffixes = ['/no_abst_all/auc_vs_acc-no_abst_norm_logits-no_abst_raw_logits',
                '/no_abst_norm_logits/auc_vs_size_all_datasets_MSP',
                '/no_abst_raw_logits/auc_vs_size_all_datasets_Max_Logit',
                '/no_abst_norm_logits/auc_vs_acc_all_datasets_MSP',
                '/no_abst_norm_logits/auc_vs_acc-no_abst_norm_logits_second_prompt-no_abst_norm_logits_first_prompt',
                '/no_abst_raw_logits/auc_vs_acc_all_datasets_Max_Logit',
                '/no_abst_raw_logits/auc_vs_acc-no_abst_raw_logits_second_prompt-no_abst_raw_logits_first_prompt',
                 ]
    file_list = [cross_group_dir + suffix for suffix in suffixes] + [input_dir + '/main_figs/' + suffix for suffix in suffixes]
    file_list += [input_dir + '/main_figs/no_abst_dataset']
    file_list += [input_dir + '/main_figs/frac-correct_vs_msp_uniform']
    file_list += [input_dir + '/main_figs/frac-correct_vs_msp_quantile']
    file_list += [input_dir + '/main_figs/calibration_table_uniform']
    file_list += [input_dir + '/main_figs/calibration_table_quantile']
    file_list += [input_dir + '/main_figs/calib_vs_acc_all_datasets']
    file_list += [input_dir + '/main_figs/calib_vs_size_all_datasets']
    file_list += [input_dir + '/main_figs/no_abst_dataset_bar']
    datasets = ['arc', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande', 'piqa', 'no_winogrande']
    middle_dirs = ['_norm_logits_first_prompt', '_norm_logits_second_prompt', '_raw_logits_first_prompt', '_raw_logits_second_prompt', '_norm_logits', '_raw_logits']
    for middle_dir in middle_dirs:
        file_list += [f'{input_dir}/main_figs/no_abst{middle_dir}/test/score_vs_conf_all_datasets']
        file_list += [f'{input_dir}/main_figs/no_abst{middle_dir}/test/harsh-score_vs_conf_all_datasets']
    for overall_cross_group_dir in ['all', 'None']:
        file_list += [cross_group_dir + f'/no_abst_{overall_cross_group_dir}/auroc_table', cross_group_dir + f'/no_abst_{overall_cross_group_dir}/auc_vs_acc-no_abst_raw_logits-no_abst_norm_logits']
        file_list += [cross_group_dir + f'/no_abst_{overall_cross_group_dir}/score_table']
        file_list += [cross_group_dir + f'/no_abst_{overall_cross_group_dir}/pct_abstained_table']
        file_list += [f'{input_dir}/{dataset}/cross_group_plots/no_abst_{overall_cross_group_dir}/{dataset}_auroc_table' for dataset in datasets]
        file_list += [f'{input_dir}/{dataset}/cross_group_plots/no_abst_{overall_cross_group_dir}/{dataset}_score_table' for dataset in datasets]
        file_list += [f'{input_dir}/{dataset}/cross_group_plots/no_abst_{overall_cross_group_dir}/{dataset}_pct_abstained_table' for dataset in datasets]
    copy_files(output_dir, file_list)

main()
