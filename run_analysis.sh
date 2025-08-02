#!/bin/bash
# Check if the correct number of arguments is provided
if [ $# -ne 3 ]; then
    echo "Incorrect number of arguments provided. Usage: ./run_analysis <input_directory> <all_figs_output_dir> <important_figs_output_dir>"
    exit 1
fi

# Assign the argument to a variable
input_dir=$1
all_figs_output_dir=$2
important_figs_output_dir=$3

# Loop over the two values of collapse_prompts
for collapse_prompts in False True
do
    # Construct the output directory name
    curr_output_dir="$all_figs_output_dir/collapse_${collapse_prompts}"

    python analysis_main.py "$curr_output_dir/main_figs" arc,hellaswag,mmlu,truthfulqa,winogrande "$collapse_prompts" "$input_dir"/*.txt
    python analysis_main.py "$curr_output_dir/arc" arc "$collapse_prompts" "$input_dir"/*.txt
    python analysis_main.py "$curr_output_dir/hellaswag" hellaswag "$collapse_prompts" "$input_dir"/*.txt
    python analysis_main.py "$curr_output_dir/mmlu" mmlu "$collapse_prompts" "$input_dir"/*.txt
    python analysis_main.py "$curr_output_dir/truthfulqa" truthfulqa "$collapse_prompts" "$input_dir"/*.txt
    python analysis_main.py  "$curr_output_dir/winogrande" winogrande "$collapse_prompts" "$input_dir"/*.txt
    python copy_important_figs.py $all_figs_output_dir $important_figs_output_dir

    # Only run statistical tests for AUROC stuff, which uses collapse_prompts=False
    if [ "$collapse_prompts" = "False" ]; then
        echo -e "\nDoing statistical tests...\n"
        for option in 1 2
	do
            python statistical_tests.py -o "$option" -d "$input_dir"
        done
    fi
done
