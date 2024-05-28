#!/bin/bash

# Check if exactly three arguments are provided
if [ $# -ne 3 ]; then
    echo "Incorrect number of arguments provided. Usage: ./do_post_processing <directory> <collapse_prompts> <incl_unparseable>"
    exit 1
fi

# Assign the arguments to variables
dir=$1
collapse_prompts=$2
incl_unparseable=$3

echo -e "collapse_prompts: $collapse_prompts, incl_unparseable: $incl_unparseable\n"

output_dir="figs_collapse-prompts-${collapse_prompts}_incl-unparseable-${incl_unparseable}"
echo -e "\nMaking figures...\n"

# We decided to not use the data where we give the LLM the option to abstain
for abstain in "no_abst"; do    
    python plot_data.py $output_dir/main_figs $incl_unparseable $collapse_prompts arc,hellaswag,mmlu,truthfulqa,winogrande $dir/*${abstain}*.txt
    python plot_data.py $output_dir/arc $incl_unparseable $collapse_prompts arc $dir/*${abstain}*.txt
    python plot_data.py $output_dir/hellaswag $incl_unparseable $collapse_prompts hellaswag $dir/*${abstain}*.txt
    python plot_data.py $output_dir/mmlu $incl_unparseable $collapse_prompts mmlu $dir/*${abstain}*.txt
    python plot_data.py $output_dir/truthfulqa $incl_unparseable $collapse_prompts truthfulqa $dir/*${abstain}*.txt
    python plot_data.py $output_dir/winogrande $incl_unparseable $collapse_prompts winogrande $dir/*${abstain}*.txt
done

echo -e "\nCopying important figures...\n"
python copy_important_figs.py $output_dir paper_figs

echo -e "\nDoing statistical tests...\n"
for option in {1..4}; do
    python statistical_tests.py -o $option -d $dir -i $incl_unparseable
done
