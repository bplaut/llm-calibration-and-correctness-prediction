Before attempting to run this code, make sure you have text generation with Hugging Face set up: https://huggingface.co/docs/transformers/llm_tutorial

# Generating text and running Q&A tests
There are two main Python files:
1. generate_text.py, which uses the Hugging Face interface to generate text with an LLM. This file can be called directly by command-line, but for our experiments it is only called by take_qa_test.py.
2. take_qa_test.py, which runs a multiple choice Q&A test using a Hugging Face dataset and generate_text.py.
Both files support the same command line arguments (shown below), although some arguments are only relevant for one file. For example, --dataset is only used for take_qa_test.py.

```
usage: generate_text.py/take_qa_test_<first/second>_prompt.py [-h] -m MODEL [-p PROMPTS] [-n MAX_NEW_TOKENS] [-k NUM_TOP_TOKENS] [-c] [-s]
                        [-r NUM_RESPONSES] [-i] [-d DATASET] [-q QUESTION_RANGE] [-b BATCH_SIZE]
                        [--abstain_option ABSTAIN_OPTION]

Perform text generation and Q&A tasks via Hugging Face models.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Which LLM to use. Check this file for currently supported options and/or add
                        your own.
  -p PROMPTS, --prompts PROMPTS
                        List of prompts, separated by |. For example "Hello my name is Ben|What a
                        time to be alive". If not provided, you will be asked for a prompt by command
                        line.
  -n MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        Number of new tokens to generate on top of the prompt
  -k NUM_TOP_TOKENS, --num_top_tokens NUM_TOP_TOKENS
                        For each token, print out the top candidates considered by the model and
                        their probabilities
  -c, --completion_mode
                        Use traditional auto-complete mode, rather than user-assistant chat
  -s, --do_sample       Should we sample from the probability distribution, or greedily pick the most
                        likely token?
  -r NUM_RESPONSES, --num_responses NUM_RESPONSES
                        Number of responses to generate per prompt. This argument is ignored for
                        greedy decoding, since that only generates one answer.
  -d DATASET, --dataset DATASET
                        The name of the Hugging Face dataset (needed for experiments and such)
  -q QUESTION_RANGE, --question_range QUESTION_RANGE
                        When running a Q&A test, what range of questions should we test? Format is
                        "-q startq-endq", 0 indexed. For example, "-q 0-100".
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Maximum number of prompts to batch together. Only used for experiments
  -a ABSTAIN_OPTION, --abstain_option ABSTAIN_OPTION
                        When running a Q&A test, should we add an option that says "I don't know"?
  -g PROMPT_PHRASING, --prompt_phrasing PROMPT_PHRASING
                        When running a Q&A test, which of the two prompt phrasings should we use? 0
                        or 1
```

# Post-processing Q&A results
There are also some files for post-processing. The first is plot_data.py, which has the following usage:
```
python plot_data.py <output_directory> <incl_unparseable> <collapse_prompts> <dataset1,dataset2,...> <data_file1> [<data_file2> ...]
```
If incl_unparseable=True, unparseable responses are counted as wrong. Otherwise, we exclude them. If collapse_prompts=True, we group the data from the two prompt phrasings together. We set collapse_prompts=False for the AUROC analysis (because it's nonlinear), but set collapse_prompt=True for the score plots (because those are linear).

There is also statistical_tests.py, which computes the p-values and has the following usage:
```
python statistical_tests.py [-h] --option OPTION --incl_unparseable INCL_UNPARSEABLE --input_dir INPUT_DIR
```
The OPTION parameter determines which tests are run and the INPUT_DIR tells the script where the data files are. See statistical_tests.py for more details.

Lastly, results_analysis.ipynb groups the p-values to create the tables in the paper.

# Batching scripts

Finally, it is tedious to call these python files individually for all the combinations of experiments and plots we want to run. For this reason, we have the following two scripts:
1. run_qa_tests.sh, which calls take_qa_test.py (which in turn calls generate_text.py). Usage:
```
./run_qa_tests.sh <comma-separated model names> <comma-separated dataset names> <comma-separated question ranges> prompt_phrasing abstain_option
```
For example,
```
./run_qa_tests.sh Mistral,Llama-13b,Llama-70b arc,truthfulqa 0-500,500-1000 0 False
```
2. do_post_processing.sh, which calls plot_data.py, copy_important_figs.py, and statistal_tests.py. Usage:
```
./do_post_processing <directory> <collapse_prompts> <incl_unparseable>
```
For example,
```
./do_post_processing results False True
```
Currently, results_analysis.ipynb is not called by the scripts and must be run separately.

# Resource requirements
We used NVIDIA RTX A6000 GPUs for our experiments, which has 48GB RAM. If you are using a GPU with less RAM, you may need to reduce the batch sizes in run_qa_tests.sh. Storing the models on disk also takes a lot of space, with the smallest (Yi 6B) taking up 12 GB, and the largest (Llama 70B) taking up 129 GB. With our setup, it took about 2 weeks to run all of the experiments from start to finish: ten models X five datasets X 6000 questions X two prompt phrasings X {no abstain option, yes abstain option}.