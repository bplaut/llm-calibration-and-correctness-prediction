from openai import OpenAI
from dotenv import load_dotenv
import math
import os
from generate_text import Generator
from string import ascii_uppercase

class OpenAIGenerator(Generator):
    def __init__(self, args):
        print("Setting up OpenAI text generator...")
        load_dotenv()
        self.client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
        self.args = args

    def generate(self, prompts):
        assert len(prompts) == 1  # not worth doing batching
        full_name = (
            'gpt-3.5-turbo-0125' if self.args['model'] == 'gpt-3.5-turbo' else
            'gpt-4-0613' if self.args['model'] == 'gpt-4' else
            'gpt-4-turbo-2024-04-09' if self.args['model'] == 'gpt-4-turbo' else
            'gpt-4o-2024-11-20' if self.args['model'] == 'gpt-4o' else
            self.args['model']
        )
        rsp = self.client.chat.completions.create(
            model=full_name,
            top_logprobs=20, # max allowed logprobs
            seed=2549900867,
            top_p=0,
            max_tokens=self.args['max_new_tokens'],
            logprobs=True,
            messages=[{"role": "user", "content": prompts[0]}],
        )
        choice = rsp.choices[0]
        text_output = [choice.message.content]
        token_output = [[tok.token for tok in choice.logprobs.content]]
        scores = [math.exp(tok.logprob) for tok in choice.logprobs.content]

        # cache first‑token top‑k distribution for later confidence calc
        self.first_token_probs = {entry.token: math.exp(entry.logprob) for entry in choice.logprobs.content[0].top_logprobs}
        self.print_output(prompts, text_output, token_output, scores)
        return text_output, token_output, scores

    def print_output(self, prompts, text_outputs, token_outputs, scores):
        print(f'PROMPT: "{prompts[0]}"')
        tokens = token_outputs[0]
        if self.args['num_top_tokens'] == 1:
            max_token_idx_len = len(str(len(tokens))) # most number of digits for token idx
            for j in range(len(tokens)):
                token = tokens[j]
                score = scores[j]
                idx_str = str(j).zfill(max_token_idx_len) # pad with 0s for prettiness
                print(f"Token {idx_str} | {round(score, 4)} | {token}")
        elif self.args['num_top_tokens'] > 1:
            # sort first token probs and print both key and value but only top num_top_tokens
            sorted_first_token_probs = sorted(self.first_token_probs.items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop first tokens: {', '.join([f'{repr(token)}: {round(prob, 4)}' for token, prob in sorted_first_token_probs[:self.args['num_top_tokens']]])}\n")
            

    def compute_confidence_levels(self, text_outputs, token_outputs, scores, choices, normalize=True):
        cand_letters = ascii_uppercase[:len(choices[0])]
        vals = {letter: self.first_token_probs.get(letter, 0) for letter in cand_letters}
        best_letter = max(vals, key=vals.get)
        return [list(vals.values())], [best_letter]
