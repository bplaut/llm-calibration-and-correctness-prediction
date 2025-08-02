import os
import shutil
import sys
import re

def copy_files(output_directory, filepaths, datasets):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for dataset in datasets:
        dataset_dir = os.path.join(output_directory, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

    for filepath in filepaths:
        if os.path.exists(filepath):
            base_name = os.path.basename(filepath)
            new_path = os.path.join(output_directory, base_name)
            for dataset in datasets:
                if dataset in filepath:
                    new_path = os.path.join(output_directory, dataset, base_name)
            shutil.copy(filepath, new_path)
            print(f"* Successfully copied {filepath}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python copy_important_figs.py <input_dir> <output_directory>")
        sys.exit(1)
    input_dir = sys.argv[1]
    input_subdir_False = input_dir + '/collapse_False'
    input_subdir_True = input_dir + '/collapse_True'
    output_dir = sys.argv[2]
    file_list = []
    dir_list = []
    datasets = ['arc', 'hellaswag', 'mmlu', 'truthfulqa', 'winogrande']
    dataset_dirs = datasets + ['main_figs']

    # AUROC: use collapse_prompts=False
    file_list.append(input_subdir_False + '/main_figs/dataset_table.tex')
    for dataset in dataset_dirs:
        main_dir = input_subdir_False + '/' + dataset
        dir_list.append(main_dir + '/mega_group_plots')
        dir_list.append(main_dir + '/pairwise_group_plots')

    # Q&A with abstention and calibration: use collapse_prompts=True
    for dataset in dataset_dirs:
        main_dir = input_subdir_True + '/' + dataset
        table_prefix = '' if dataset == 'main_figs' else dataset + '_'
        for conf_type in ['max', 'margin']:
            for measure_type in ['score', 'pct_abstained']:
                file_list.append(main_dir + f'/{measure_type}_{table_prefix}{conf_type}_k20_table.tex')
        dir_list.append(main_dir + '/num_train_plots')
        plot_dir = main_dir + '/single_group_plots'
        if os.path.exists(plot_dir):
            for directory in os.listdir(plot_dir):
                new_path = os.path.join(plot_dir, directory)
                if os.path.isdir(new_path):
                    dir_list.append(new_path)
                    dir_list.append(new_path + '/test') # Add the score plots from the test directory
                        
    # Add all files in the dir_list (and subdirectories). However, only include files that either (1) don't have the k[number] pattern, i.e., don't relate to training data (like AUROC), or (2) have num_train = 20.
    for directory in dir_list:
        for root, _, files in os.walk(directory):
            for f in files:
                pattern = r'^(?!.*k\d).*$|^(?=.*k20(?!\d)).*$'
                if re.match(pattern, f) and ('collapse_False' in directory or 'auc' not in f):
                    # only include AUROC files for collapse_False
                    file_list.append(os.path.join(root, f))
                    if 'auc' in f:
                        print(f"Added {os.path.join(root, f)} to file list")
                    
    copy_files(output_dir, file_list, datasets)

main()
