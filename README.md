Before attempting to run this code, make sure you have text generation with Hugging Face set up: https://huggingface.co/docs/transformers/llm_tutorial

# Generating text and running Q&A tests
There are two main Python files:
1. generate_text.py, which uses the Hugging Face interface to generate text with an LLM. This file can be called directly by command-line, but for our experiments it is only called by take_qa_test.py.
2. take_qa_test.py, which runs a multiple choice Q&A test using a Hugging Face dataset and generate_text.py.
Both files support the same command line arguments (shown below), although some arguments are only relevant for one file. For example, --dataset is only used for take_qa_test.py.

```
usage: generate_text.py/take_qa_tests.py [-h] -m MODEL [-p PROMPTS] [-n MAX_NEW_TOKENS] [-k NUM_TOP_TOKENS] [-c] [-s]
                       [-r NUM_RESPONSES] [-d DATASET] [-q QUESTION_RANGE] [-b BATCH_SIZE] [-a ABSTAIN_OPTION]
                       [-g PROMPT_PHRASING] [-f FEW_SHOT_NUMBER]

Perform text generation and Q&A tasks via Hugging Face models.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Which LLM to use. Check this file for currently supported options and/or add your own.
  -p PROMPTS, --prompts PROMPTS
                        List of prompts, separated by |. For example "This is a prompt|What a time to be
                        alive".
  -n MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        Number of new tokens to generate on top of the prompt
  -k NUM_TOP_TOKENS, --num_top_tokens NUM_TOP_TOKENS
                        For each token, print out the top candidates considered by the model and their
                        probabilities
  -c, --completion_mode
                        Use traditional auto-complete mode, rather than user-assistant chat
  -s, --do_sample       Should we sample from the probability distribution, or greedily pick the most likely token?
  -r NUM_RESPONSES, --num_responses NUM_RESPONSES
                        Number of responses to generate per prompt. This argument is ignored for greedy decoding,
                        since that only generates one answer.
  -d DATASET, --dataset DATASET
                        The name of the Hugging Face dataset (needed for experiments and such)
  -q QUESTION_RANGE, --question_range QUESTION_RANGE
                        When running a Q&A test, what range of questions should we test? Format is "-q startq-
                        endq", 0 indexed. For example, "-q 0-100".
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Maximum number of prompts to batch together. Only used for experiments
  -a ABSTAIN_OPTION, --abstain_option ABSTAIN_OPTION
                        When running a Q&A test, should we add an option that says "I don't know"?
  -g PROMPT_PHRASING, --prompt_phrasing PROMPT_PHRASING
                        When running a Q&A test, which of the two prompt phrasings should we use? 0 or 1
  -f FEW_SHOT_NUMBER, --few_shot_number FEW_SHOT_NUMBER
                        When running a Q&A test, how many in-context examples to provide?
```

# Running analysis
The main python file for the analysis is (unsurprisingly) analysis_main.py, which has the following usage:
```
python analysis_main.py <output_directory> <dataset1,dataset2,...> <collapse_prompts> <data_file1> [<data_file2> ...]
```
If collapse_prompts=True, we group the data from the two prompt phrasings together. We set collapse_prompts=False for the AUROC analysis (because it's nonlinear), but set collapse_prompt=True for the Q&A-with-abstention analysis (because those are linear). We also collapse prompts for the calibration analysis because it's kind of weird to make a separate calibration curve for each plot and average them. You probably don't need to worry about this because you should probably just use the bash scripts below, which will automatically use collapse_prompts=True/False in the appropriate places.

The files plotting_functions.py and utils.py are called by analysis_main.py. (Note that utils.py is also used briefly in generate_text.py.)

There is also statistical_tests.py, which computes the p-values and has the following usage:
```
python statistical_tests.py [-h] --option OPTION --input_dir INPUT_DIR
```
The OPTION parameter determines which tests are run and the INPUT_DIR tells the script where the data files are. See statistical_tests.py for more details.

The analysis_main.py will generate a lot of figures, and we aren't using them all in the paper, so copy_important_figs.py automatically copies the relevant figures into a specified directory.

# Batching scripts

It is tedious to call these python files individually for all the experiments and analyses we want to run. For this reason, we have the following two bash scripts:
1. run_qa_tests.sh, which calls take_qa_test.py (which in turn calls generate_text.py). Usage:
```
./run_qa_tests.sh <comma-separated model names> <comma-separated dataset names> <comma-separated question ranges> prompt_phrasing few_shot
```
For example, to run all of the experiments for the first prompt phrasing and zero-shot, the command would be
```
./run_qa_tests.sh Falcon-7b,Falcon-40b,Llama3-8b,Llama3-70b,Llama3.1-8b,Llama3.1-70b,Llama-7b,Llama-70b,Mistral,Mixtral,Solar,Yi-6b,Yi-34b,gpt-3.5-turbo,gpt-4o arc,hellaswag,mmlu,truthfulqa,winogrande 0-1000,1000-2000,2000-3000,3000-4000,4000-5000,5000-6000 0 False
```
To run the second prompt, one would replace the final 0 with 1. To use five-shot prompting, one would replace the False with True.

If you get an out-of-memory error, try reducing the batch sizes in run_qa_tests.sh.

2. run_analysis.sh, which calls analysis_main.py, copy_important_figs.py, and statistal_tests.py. Usage:
```
./run_analysis <input_dir> <all_figs_output_dir> <important_figs_output_dir>
```
For example, the command
```
./run_analysis chat_results all_figs important_figs
```
will take as input the results files from main_results, save all figures to the all_figs directory, and copy the important figures (which mostly means the figures used in the paper) to important_figs. Relevant result files directories:
- chat_results: zero shot results for chat models (used for primary analysis)
- 5shot_chat_results: 5-shot results for chat models
- base_results: zero shot results for base (i.e., non-fine-tuned) models
- 5shot_base_results: 5-shot results for base models

# Resource requirements
We used NVIDIA RTX A6000 GPUs for our experiments, which has 48GB RAM. If you are using a GPU with less RAM, you may need to reduce the batch sizes in run_qa_tests.sh. Storing the models on disk also takes a lot of space, with the smallest (Falcon 7B) taking up 14 GB, and the largest (Llama 2 70B) taking up 224 GB. Running the experiments for the HuggingFace models took about 2000 GPU-hours time using A6000s, roughly equally split between the chat LLMs and base LLMs. Running the experiments for the OpenAI models cost about $120.