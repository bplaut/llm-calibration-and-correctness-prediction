#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage format: ./run_qa_tests.sh model1,model2 dataset1,dataset2,dataset3 question_range1,question_range2 prompt_phrasing abstain_option"
    echo "Example usage: ./run_qa_tests.sh Mistral,Llama-13b,Llama-70b arc,truthfulqa 0-500,500-1000 0 False"
    exit 1
fi

# Read the argument options from the command line input and split them into arrays
IFS=',' read -r -a model_options <<< "$1"
IFS=',' read -r -a dataset_options <<< "$2"
IFS=',' read -r -a question_ranges <<< "$3"
prompt_phrasing="$4"
abstain_option="$5"

# Function to determine batch_size based on model and dataset names
get_batch_size() {
    local model_name="$1"
    local dataset_name="$2"
    local batch_size

    # remove -raw from the model name if it exists
    model_name="${model_name/-raw/}"

    case "$model_name" in
        "Llama-70b")
            batch_size=16
            ;;
        "Llama-7b")
            batch_size=40
            ;;
        "Llama-13b")
            batch_size=22
            ;;
        "Falcon-40b")
            batch_size=10
            ;;
	"Falcon-7b")
	    batch_size=168
	    ;;
	"Yi-6b"|"Yi-34b") # For some reason, this crashes for batch_size > 1
	    batch_size=1
	    ;;
        "Mistral"|"Zephyr"|"Solar")
            batch_size=128
            ;;
        *)
            batch_size=63 # Default value
            ;;
    esac

    # For some datasets, adjust batch sizes. +2 to ensure that it doesn't go to 0
    if [ "$dataset_name" = "mmlu" ]; then
        batch_size=$(( (batch_size + 2) / 3 ))
    fi
    if [ "$dataset_name" = "piqa" ]; then
	    batch_size=$(( (batch_size + 2) / 3 ))
    fi

    echo "$batch_size"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop through each combination of model, dataset, and question_range
for question_range in "${question_ranges[@]}"
do
    for dataset in "${dataset_options[@]}"
    do
        for model in "${model_options[@]}"
        do
            # Determine batch_size based on the model and dataset
            batch_size=$(get_batch_size "$model" "$dataset")

            # Define log file name
            log_file="logs/${model}_${dataset}_${question_range}_abstain-option-${abstain_option}_prompt-phrasing-${prompt_phrasing}_log.txt"

            # Running the command with the arguments
            echo -e "\nRunning take_qa_test.py with arguments: --model=$model --dataset=$dataset --question_range=$question_range --batch_size=$batch_size --abstain_option=$abstain_option --prompt_phrasing=$prompt_phrasing"
            python take_qa_test.py --model="$model" --dataset="$dataset" --question_range="$question_range" --batch_size="$batch_size" --abstain_option="$abstain_option" --prompt_phrasing="$prompt_phrasing" --max_new_tokens=100 --num_top_tokens=1 &> "$log_file"
        done
    done
done

echo "Script completed."
