from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import torch as t
from utils import str_to_bool
from string import ascii_uppercase

class Generator(object):
    def __init__(self, args):
        model_name_map = {'Mistral-base':'mistralai/Mistral-7B-v0.1',
                          'Mistral':'mistralai/Mistral-7B-Instruct-v0.2',
                          'Mixtral-base':'mistralai/Mixtral-8x7B-v0.1',
                          'Mixtral':'mistralai/Mixtral-8x7B-Instruct-v0.1',
                          'Llama2-7b-base':'meta-llama/Llama-2-7b-hf',
                          'Llama2-7b':'meta-llama/Llama-2-7b-chat-hf',
                          'Llama2-13b-base':'meta-llama/Llama-2-13b-hf',
                          'Llama2-13b':'meta-llama/Llama-2-13b-chat-hf',
                          'Llama2-70b-base':'meta-llama/Llama-2-70b-hf',
                          'Llama2-70b':'meta-llama/Llama-2-70b-chat-hf',
                          'Falcon-7b-base':'tiiuae/falcon-7b',
                          'Falcon-7b':'tiiuae/falcon-7b-instruct',
                          'Falcon-40b-base':'tiiuae/falcon-40b',
                          'Falcon-40b':'tiiuae/falcon-40b-instruct',
                          'Solar-base':'upstage/SOLAR-10.7B-v1.0',
                          'Solar':'upstage/SOLAR-10.7B-Instruct-v1.0',
                          'Yi-34b':'01-ai/Yi-34B-Chat',
                          'Yi-6b':'01-ai/Yi-6B-Chat',
                          'Yi-34b-base':'01-ai/Yi-34B',
                          'Yi-6b-base':'01-ai/Yi-6B',
                          'Llama3-8b':'meta-llama/Meta-Llama-3-8B-Instruct',
                          'Llama3-70b':'meta-llama/Meta-Llama-3-70B-Instruct',
                          'Llama3.1-8b':'meta-llama/Llama-3.1-8B-Instruct',
                          'Llama3.1-70b':'meta-llama/Llama-3.1-70B-Instruct',
                          'Llama3-8b-base':'meta-llama/Meta-Llama-3-8B',
                          'Llama3-70b-base':'meta-llama/Meta-Llama-3-70B',
                          'Llama3.1-8b-base':'meta-llama/Llama-3.1-8B',
                          'Llama3.1-70b-base':'meta-llama/Llama-3.1-70B',
        }
        if args['model'] not in model_name_map:
            raise Exception("Unrecognized model name. Check model_name_map")
        else:
            model_name = model_name_map[args['model']]
        if args['model'].endswith('-base'):
            args['completion_mode'] = True # base models don't have a chat mode
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.args = args
        self._letter_token_ids = self._build_letter_token_id_map()

    def _build_letter_token_id_map(self):
        """Return dict mapping 'A'... 'Z' to list of token IDs whose decoded
        form (after stripping whitespace and BPE markers) equals that letter."""
        mapping = {L: [] for L in ascii_uppercase}
        for tok_id in range(self.tokenizer.vocab_size):
            decoded = self.tokenizer.decode(tok_id)
            decoded = decoded.replace('▁', '').replace('Ġ', '').strip()
            if decoded in mapping:
                mapping[decoded].append(tok_id)
        return mapping

    def compute_confidence_levels(self, text_outputs, token_outputs, scores, choices, normalize=True):
        """For each example i, return `(confidence_i, predicted_letter_i)` where
        confidence_i is the distribution of probabilities (or unnormalised logit) of 
        each of the candidate letters. predicted_letter_i is the argmax letter
        """
        num_prompts = len(choices)
        confs, preds = [], []
        for i in range(num_prompts):
            if len(token_outputs[i]) == 1: # single token: only one choice
                first_token_scores = scores[0, i]
            elif self.args['model'] in ['Llama2-7b','Llama2-70b','Falcon-40b','Falcon-40b-base', 'Yi-6b-base','Yi-34b-base']: # These models always start with a dummy token
                first_token_scores = scores[1, i]
            else:
                first_token_scores = scores[0, i]
                
            if normalize:
                first_token_scores = t.softmax(first_token_scores, dim=-1)
            cand_letters = ascii_uppercase[:len(choices[i])]
            letter_vals = {}
            for L in cand_letters:
                ids = self._letter_token_ids[L]
                if normalize:
                    letter_vals[L] = first_token_scores[ids].sum().item()
                else:
                    letter_vals[L] = t.logsumexp(first_token_scores[ids], dim=0).item()
                
            best_letter = max(letter_vals, key=letter_vals.get)
            confs.append(list(letter_vals.values()))
            preds.append(best_letter)
        return confs, preds

    def prepare_for_chat(self, prompts):
        if self.args['completion_mode'] or ('Falcon' in self.args['model']):
            return prompts # Base models (and for some reason, Falcon chat models)don't use chat templates
        else:
            chats = [[{"role": "user", "content": p}] for p in prompts]
            return [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True, return_tensors="pt") for c in chats]

    def print_output(self, prompts, text_outputs, token_outputs, scores):
        print('\n')
        for i in range(len(text_outputs)):
            print('PROMPT %d: "%s"\n' % (i+1, prompts[i]))
            print('OUTPUT %d: "%s"\n' % (i+1, text_outputs[i]))
            pad_token_idxs = (token_outputs[i] == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            first_pad_idx = pad_token_idxs[0].item() if len(pad_token_idxs) > 0 else len(token_outputs[i])

            if self.args['num_top_tokens'] > 0:
                for j in range(len(token_outputs[i])):
                    if self.tokenizer.decode(token_outputs[i][j]) == self.tokenizer.pad_token and j > 0:
                        # If we have prompts/responses of different lengths, some will get padded. Ignore those tokens, but always print at least one token (this is helpful for debugging if e.g. the model only outputs a pad token)
                        break
                    
                    # scores has shape (response_length, num_prompts, vocab_size)
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
        prompts = self.prepare_for_chat(prompts)
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        output = self.model.generate(**model_inputs, max_new_tokens=self.args['max_new_tokens'], do_sample=False, output_scores=True, num_return_sequences=1, return_dict_in_generate=True, renormalize_logits=False)
        token_inputs = model_inputs['input_ids'] if 'Yi' in self.args['model'] else model_inputs # Yi models have a different format here for some reason
        token_outputs = [output.sequences[i][len(token_inputs[i]):] for i in range(len(output.sequences))] # non-prompt part of the output, tokenized
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
    parser.add_argument('-p', '--prompts', type=str, help='List of prompts, separated by |. For example "This is a prompt|What a time to be alive".', default=None)
    parser.add_argument('-n', '--max_new_tokens', type=int, help='Number of new tokens to generate on top of the prompt', default=100)
    parser.add_argument('-k', '--num_top_tokens', type=int, help='For each token, print out the top candidates considered by the model and their probabilities', default=0)
    parser.add_argument('-c', '--completion_mode', action="store_true", help='Use traditional auto-complete mode, rather than user-assistant chat', default=False)
    parser.add_argument('-d', '--dataset', type=str, default=None, help='The name of the Hugging Face dataset (needed for experiments and such)')
    parser.add_argument('-q', '--question_range', type=str, help='When running a Q&A test, what range of questions should we test? Format is "-q startq-endq", 0 indexed. For example, "-q 0-100".', default=None)
    parser.add_argument('-b', '--batch_size', type=int, help='Maximum number of prompts to batch together. Only used for experiments', default=1)
    parser.add_argument('-g', '--prompt_phrasing', type=int, help='When running a Q&A test, which of the two prompt phrasings should we use? 0 or 1', default=0)
    parser.add_argument('-f', '--few_shot_number', type=int, help='When running a Q&A test, how many in-context examples to provide?', default=0)
    return dict(vars(parser.parse_args())) # dictionaries are easier to manipulate sometimes

def t_to_str(T, precision=3):
    # Print tensors like python lists
    s = str(T)
    last_bracket_idx = s.rfind(']')
    if last_bracket_idx != -1:
        s = s[:last_bracket_idx + 1] # remove everything after the last bracket
    else:
        s = s[:s.rfind(',')] # singleton tensor. Remove last comma and afterwards
    s = s.replace("tensor(", "")
    s = s.replace("\n", "")
    s = s.replace("    ", "")
    target_len = precision + 2 # e.g. 0.534
    return s + '0' * (target_len - len(s)) if '.' in s else s # pad with 0s if decimal
    
def main():
    t.set_printoptions(sci_mode=False, precision=3)
    args = parse_args() 
    generator = Generator(args)

    if args['prompts'] == None:
        prompts = [input("\nEnter a prompt:\n")]
        print('\n')
    else:
        prompts = args['prompts'].split('|')

    generator.generate(prompts)
        
if __name__ == '__main__':
    main()
