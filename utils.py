import sys

def parse_file_name(file_name, collapse_prompts=False):
    # filename looks like <dataset>_<model>-q<startq>to<endq>_<group>.txt  
    second_half = file_name[file_name.find('to'):]
    parts = file_name.split('_')
    dataset = parts[0]
    model = parts[1].split('-q')[0]
    group = second_half[second_half.find('_')+1:-4] # remove initial underscore and .txt
    if collapse_prompts:
        group = group.replace('_first_prompt','').replace('_second_prompt', '')
    return dataset, model, group

def parse_data(file_path, incl_unparseable):
    labels = []
    conf_levels = []
    total_qs = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # skip header line                                                                          
                # skip Abstained answers, which don't affect the score or auc                               
                # skip Unparseable lines if incl_unparseable is False                                       
                if parts[0] in ("Correct", "Wrong") or (incl_unparseable and parts[0] == "Unparseable"):
                    labels.append(1 if parts[0] == "Correct" else 0)
                    conf_levels.append(float(parts[1]))
                if parts[0] in ("Correct", "Wrong", 'Abstained') or (incl_unparseable and parts[0] == "Unparseable"):
                    # Abstentions don't affect the score, but we still want them for normalization          
                    total_qs += 1
    except IOError:
        print(f"Error opening file: {file_path}")
        sys.exit(1)
    return labels, conf_levels, total_qs

def expand_model_name(name):
    base = name.split('-raw')[0]
    base_expanded = ('Mistral 7B' if base == 'Mistral' else
                     'Mixtral 8x7B' if base == 'Mixtral' else
                     'SOLAR 10.7B' if base == 'Solar' else
                     'Llama2 13B' if base == 'Llama-13b' else
                     'Llama2 7B' if base == 'Llama-7b' else
                     'Llama2 70B' if base == 'Llama-70b' else
                     'Yi 6B' if base == 'Yi-6b' else
                     'Yi 34B' if base == 'Yi-34b' else
                     'Falcon 7B' if base == 'Falcon-7b' else
                     'Falcon 40B' if base == 'Falcon-40b' else base)
    return base_expanded + ' Raw' if name.endswith('-raw') else base_expanded

def expand_label(label):
        return ('Confidence Threshold' if label == 'conf' else
                'Score (Balanced)' if label == 'score' else
                'Score (Conservative)' if label == 'harsh-score' else
                'Model Size (billions of parameters)' if label == 'size' else
                'AUROC' if label == 'auc' else
                'Q&A Accuracy' if label == 'acc' else label)

# Each model name is of the form "<model_series> <size>B. Mixtral is a slight exception 
def model_series(name):
    return expand_model_name(name).split(' ')[0]

def model_size(name):
    full_name = expand_model_name(name)
    size_term = full_name.split(' ')[1]
    end_of_size_term = size_term.rfind('B')
    return 46.7 if 'Mixtral' in name else float(size_term[:end_of_size_term])

def group_label(group):
    logit_type = 'MSP' if group.startswith('no_abst_norm_logits') else 'Max Logit' if group.startswith('no_abst_raw_logits') else group
    prompt = ', first phrasing' if group.endswith('first_prompt') else ', second phrasing' if group.endswith('second_prompt') else ''
    return logit_type, prompt

def make_pct(x):
    return 100*x

def parse_group_name(group, collapse_prompts=False):
    # Each group name has the form <yes/no>_abst_<raw/norm>_logits_<first/second>_prompt
    # Later on, we might create some merged groups, but we'll only call this fn on initial groups
    parts = group.split('_')
    abst_type = parts[0] + '_' + parts[1]
    logit_type = parts[2] + '_' + parts[3]
    prompt_type = parts[4] + '_' + parts[5] if not collapse_prompts else None
    return (abst_type, logit_type, prompt_type)

def format_dataset_name(dataset):
    return ('ARC-Challenge' if dataset == 'arc' else
            'HellaSwag' if dataset == 'hellaswag' else
            'MMLU' if dataset == 'mmlu' else
            'PIQA' if dataset == 'piqa' else
            'TruthfulQA' if dataset == 'truthfulqa' else
            'WinoGrande' if dataset == 'winogrande' else dataset)

def plot_style_for_group(group):
    return (('teal', 's', 'black') if 'first_prompt' in group else
            ('gold', '^', 'black') if 'second_prompt' in group else
            ('deepskyblue', 'o', 'tab:red') if 'norm' in group else
            ('mediumpurple', 'D', 'tab:orange') if 'raw' in group else ('#1f77b4', 'o', 'red'))

def str_to_bool(s):
    if s.lower() in ('true', 'yes', 'y', '1'):
        return True
    elif s.lower() in ('false', 'no', 'n', '0'):
        return False
    else:
        raise Exception("Unrecognized boolean string")
