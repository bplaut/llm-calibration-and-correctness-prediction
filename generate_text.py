from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import torch as t
from utils import str_to_bool
from string import ascii_uppercase

class Generator(object):
    def __init__(self, args):
        model_name_map = {'Mistral-raw':'mistralai/Mistral-7B-v0.1',
                          'Mistral':'mistralai/Mistral-7B-Instruct-v0.2',
                          'Mixtral-raw':'mistralai/Mixtral-8x7B-v0.1',
                          'Mixtral':'mistralai/Mixtral-8x7B-Instruct-v0.1',
                          'Zephyr':'HuggingFaceH4/zephyr-7b-beta',
                          'gpt2':'gpt2',
                          'Llama-7b-raw':'meta-llama/Llama-2-7b-hf',
                          'Llama-7b':'meta-llama/Llama-2-7b-chat-hf',
                          'Llama-13b-raw':'meta-llama/Llama-2-13b-hf',
                          'Llama-13b':'meta-llama/Llama-2-13b-chat-hf',
                          'Llama-70b-raw':'meta-llama/Llama-2-70b-hf',
                          'Llama-70b':'meta-llama/Llama-2-70b-chat-hf',
                          'Falcon-7b-raw':'tiiuae/falcon-7b',
                          'Falcon-7b':'tiiuae/falcon-7b-instruct',
                          'Falcon-40b-raw':'tiiuae/falcon-40b',
                          'Falcon-40b':'tiiuae/falcon-40b-instruct',
                          'Solar-raw':'upstage/SOLAR-10.7B-v1.0',
                          'Solar':'upstage/SOLAR-10.7B-Instruct-v1.0',
                          'Yi-34b':'01-ai/Yi-34B-Chat',
                          'Yi-6b':'01-ai/Yi-6B-Chat',
                          'Yi-34b-raw':'01-ai/Yi-34B',
                          'Yi-6b-raw':'01-ai/Yi-6B',
                          'Llama3-8b':'meta-llama/Meta-Llama-3-8B-Instruct',
                          'Llama3-70b':'meta-llama/Meta-Llama-3-70B-Instruct',
        }
        if args['model'] not in model_name_map:
            raise Exception("Unrecognized model name. Check model_name_map")
        else:
            model_name = model_name_map[args['model']]
        if 'raw' in args['model']:
            args['completion_mode'] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.args = args

        self.num_responses = 1 if not self.args['do_sample'] else self.args['num_responses']

    def compute_confidence_levels(self, text_outputs, token_outputs, scores, choices, normalize=True):
        # Find the max probability for the token which determines the answer      
        confidence_levels = [None] * len(text_outputs)
        for (i, response) in enumerate(text_outputs):
            num_choices = len(choices[i]) if len(choices) > i else 0
            main_targets = [c + '.' for c in ascii_uppercase][:num_choices]
            backup_targets = choices[i] + [c for c in ascii_uppercase][:num_choices]
            token_idx1 = self.token_idx_of_first_target(response, main_targets)
            token_idx2 = self.token_idx_of_first_target(response, backup_targets)
            token_idx = token_idx1 if token_idx1 != -1 else token_idx2
            (conf, _) = self.min_max_logit(scores, i, lo=token_idx, hi=token_idx+1, normalize=normalize)
            confidence_levels[i] = conf
        return confidence_levels
        
    def min_max_logit(self, scores, response_idx, lo=0, hi=None, normalize=True):
        # scores has shape (response_length, num_responses, vocab_size)
        scores = scores[lo:hi,::] 
        if len(scores) == 0: # For example, when we call this fn with lo=first_token_idx(x)=len(scores) if we don't find token x
            return (0, None)
        if normalize:
            scores = t.exp(scores) / t.sum(t.exp(scores), dim=2, keepdim=True)
        (max_logit_per_token, _) = t.max(scores, dim=2)
        (min_among_max_logits, indices) = t.min(max_logit_per_token, dim=0)
        return (min_among_max_logits[response_idx], indices[response_idx])

    def token_idx_of_first_target(self, s, targets):
        # Find the first index of a target in s. Then find the index of the corresponding token in the tokenized version of s
        target_idxs = [s.find(target) for target in targets if s.find(target) != -1]
        if len(target_idxs) > 0:
            i = min(target_idxs)
            # Find the index of the token that contains the character at index i
            tokens = self.tokenizer(s, return_offsets_mapping=True, add_special_tokens=False)
            for token_index, (start, end) in enumerate(tokens.offset_mapping):
                if start <= i < end and 'Llama3' not in self.args['model']:
                    return token_index
                elif start <= i and 'Llama3' in self.args['model']: # Llama3 models have a different offset mapping where start and end are always the same. E.g., the offset mapping will be [(0,0), (3,3)] if the first token starts at index 0 and the second at index 3
                    return token_index
        return -1

    def first_token_instance(self, token_id_seq, target_tokens):
        target_token_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
        # The first 0 index is because t.where returns a tuple with one elem per dim
        where_each_token = [t.where(token_id_seq == token)[0] for token in target_token_ids if token is not None]
        # The next 0 index is because we want the 1st index containing a target (if any exist)
        return min([w[0].item() if len(w) > 0 else len(token_id_seq) for w in where_each_token])
    
    def prepare_for_chat(self, prompts):
        if self.args['model'] in ('Falcon-7b', 'Falcon-40b'):
            return prompts # These models doesn't use chat templates
        else:
            chats = [[{"role": "user", "content": p}] for p in prompts]
            return [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, return_tensors="pt") for c in chats]

    def print_output(self, prompts, text_outputs, token_outputs, scores):
        print('\n')
        for i in range(len(text_outputs)):
            prompt_idx = i//self.num_responses
            print('PROMPT %d: "%s"\n' % (prompt_idx+1, prompts[prompt_idx]))
            print('OUTPUT %d: "%s"\n' % (i % self.num_responses + 1, text_outputs[i]))
            pad_token_idxs = (token_outputs[i] == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            first_pad_idx = pad_token_idxs[0].item() if len(pad_token_idxs) > 0 else len(token_outputs[i])

            if self.args['num_top_tokens'] > 0:
                (mm_logit,mm_logit_idx) = self.min_max_logit(scores,i,lo=0,hi=first_pad_idx,normalize=False)
                (mm_prob,mm_prob_idx) = self.min_max_logit(scores,i,lo=0,hi=first_pad_idx,normalize=True)
                print("Min max prob  =", t_to_str(mm_prob), "| Index =", t_to_str(mm_prob_idx))
                print("Min max logit =", t_to_str(mm_logit), "| Index =", t_to_str(mm_logit_idx))
                for j in range(len(token_outputs[i])):
                    if self.tokenizer.decode(token_outputs[i][j]) == self.tokenizer.pad_token:
                        # If we have prompts/responses of different lengths, some will get padded
                        break
                    
                    # scores has shape (response_length, num_responses, vocab_size) 
                    (sorted_scores, top_token_ids) = t.sort(scores[j][i], descending=True)
                    sorted_probs = t.exp(sorted_scores) / t.sum(t.exp(sorted_scores))
                    top_tokens = self.tokenizer.batch_decode(top_token_ids[:self.args['num_top_tokens']])
                    if self.args['num_top_tokens'] == 1:
                        max_token_idx_len = len(str(len(token_outputs[i])))
                        idx_str = str(j).zfill(max_token_idx_len) # pad with 0s for prettiness
                        print("Token %s |" % idx_str, t_to_str(sorted_probs[0]), '|', t_to_str(sorted_scores[0]), '|', repr(top_tokens[0]))
                    else:
                        print('\nToken %d:' % j, repr(self.tokenizer.decode(token_outputs[i][j])))
                        print("Top tokens:", top_tokens)
                        print("Top probs:", t_to_str(sorted_probs[:self.args['num_top_tokens']]))
                        print("Top logits:", t_to_str(sorted_scores[:self.args['num_top_tokens']]))
            print('\n')
            
    def generate(self, prompts):
        prompts = self.prepare_for_chat(prompts) if not self.args['completion_mode'] else prompts
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        output = self.model.generate(**model_inputs, max_new_tokens=self.args['max_new_tokens'], do_sample=self.args['do_sample'], output_scores=True, num_return_sequences=self.num_responses, return_dict_in_generate=True, renormalize_logits=False)
        token_inputs = model_inputs['input_ids'] if 'Yi' in self.args['model'] else model_inputs
        token_outputs = [output.sequences[i][len(token_inputs[i//self.num_responses]):] for i in range(len(output.sequences))] # non-prompt part of the output, tokenized. i//num_responses in the prompt index
        text_outputs = self.tokenizer.batch_decode(token_outputs, skip_special_tokens=True)

        scores = t.stack(list(output.scores), dim=0) # initially it's a tuple of tensors
        if scores.dtype != t.float32:
            print("Casting scores to float32")
            scores = scores.to(t.float32)
            
        self.print_output(prompts, text_outputs, token_outputs, scores)
        return (text_outputs, token_outputs, scores)

def parse_args():
    parser = argparse.ArgumentParser(description='Perform text generation and Q&A tasks via Hugging Face models.')
    parser.add_argument('-m', '--model', type=str, help='Which LLM to use. Check this file for currently supported options and/or add your own.',required=True)
    parser.add_argument('-p', '--prompts', type=str, help='List of prompts, separated by |. For example "Hello my name is Ben|What a time to be alive". If not provided, you will be asked for a prompt by command line.', default=None)
    parser.add_argument('-n', '--max_new_tokens', type=int, help='Number of new tokens to generate on top of the prompt', default=10)
    parser.add_argument('-k', '--num_top_tokens', type=int, help='For each token, print out the top candidates considered by the model and their probabilities', default=0)
    parser.add_argument('-c', '--completion_mode', action="store_true", help='Use traditional auto-complete mode, rather than user-assistant chat', default=False)
    parser.add_argument('-s', '--do_sample', action="store_true", help='Should we sample from the probability distribution, or greedily pick the most likely token?', default=False)
    parser.add_argument('-r', '--num_responses', type=int, help='Number of responses to generate per prompt. This argument is ignored for greedy decoding, since that only generates one answer.', default=1)
    parser.add_argument('-d', '--dataset', type=str, default=None, help='The name of the Hugging Face dataset (needed for experiments and such)')
    parser.add_argument('-q', '--question_range', type=str, help='When running a Q&A test, what range of questions should we test? Format is "-q startq-endq", 0 indexed. For example, "-q 0-100".', default=None)
    parser.add_argument('-b', '--batch_size', type=int, help='Maximum number of prompts to batch together. Only used for experiments', default=1)
    parser.add_argument('-a', '--abstain_option', type=str_to_bool, help='When running a Q&A test, should we add an option that says "I don\'t know"?', default=False)
    parser.add_argument('-g', '--prompt_phrasing', type=int, help='When running a Q&A test, which of the two prompt phrasings should we use? 0 or 1', default=0)
    parser.add_argument('-f', '--few_shot_number', type=int, help='When running a Q&A test, how many in-context examples to provide?', default=0)
    return dict(vars(parser.parse_args())) # dictionaries are easier to manipulate sometimes

def t_to_str(T):
    # Get rid of a bunch of stuff in the tensor format that I don't like
    s = str(T)
    last_bracket_idx = s.rfind(']')
    if last_bracket_idx != -1:
        s = s[:last_bracket_idx + 1] # remove everything after the last bracket
    else:
        s = s[:s.rfind(',')] # singleton tensor. Remove last comma and afterwards
    s = s.replace("tensor(", "")
    s = s.replace("\n", "")
    s = s.replace("    ", "")
    target_len = 5 # e.g. 0.534
    return s + '0' * (target_len - len(s)) if '.' in s else s # pad with 0s if decimal
    
def main():
    t.set_printoptions(sci_mode=False, precision=3)
    args = parse_args() 
    generator = Generator(args)

    if args['prompts'] == None:
        prompts = [input("\nEnter an initial prompt:\n")]
        print('\n')
    else:
        prompts = args['prompts'].split('|')
    
    generator.generate(prompts)
        
if __name__ == '__main__':
    main()
