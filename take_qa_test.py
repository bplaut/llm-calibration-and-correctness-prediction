from generate_text import Generator, t_to_str, parse_args
from openai_generate import OpenAIGenerator
import os
from datasets import load_dataset, concatenate_datasets
import random
from string import ascii_uppercase


class Test(object):
    def __init__(self, args):
        bounds = args['question_range'].split('-')
        self.start_q, self.end_q = int(bounds[0]), int(bounds[1])
        self.model = (OpenAIGenerator(args) if 'gpt' in args['model'] else Generator(args))
        self.args = args

        dset_name = args['dataset'].lower()
        dset = (
            load_dataset('Rowan/hellaswag', split='train') if dset_name == 'hellaswag' else
            concatenate_datasets([
                load_dataset('ai2_arc', 'ARC-Challenge', split=s) for s in ['train', 'test', 'validation']
            ]) if dset_name == 'arc' else
            concatenate_datasets([
                load_dataset('winogrande', 'winogrande_debiased', split=s) for s in ['train', 'validation']
            ]) if dset_name == 'winogrande' else
            load_dataset('cais/mmlu', 'all', split='test') if dset_name == 'mmlu' else
            load_dataset('truthful_qa', 'multiple_choice', split='validation') if dset_name == 'truthfulqa' else
            None
        )
        if dset is None:
            raise Exception(f"Unsupported dataset name: {dset_name}")
        self.questions = list(dset)
        random.shuffle(self.questions)
        self.end_q = min(self.end_q, len(self.questions))

        self.get_q = (
            lambda x: x['ctx'] if dset_name == 'hellaswag' else
            x['question'] if dset_name in ['arc', 'mmlu', 'truthfulqa'] else
            x['sentence'] if dset_name == 'winogrande' else None
        )
        self.get_a = (
            lambda x: self.make_index(x['label']) if dset_name == 'hellaswag' else
            self.make_index(x['answerKey'], 1) if dset_name == 'arc' else
            self.make_index(x['answer'], 1) if dset_name == 'winogrande' else
            x['answer'] if dset_name == 'mmlu' else
            x['mc1_targets']['labels'].index(1) if dset_name == 'truthfulqa' else None
        )
        self.get_choices = (
            lambda x: x['endings'] if dset_name == 'hellaswag' else
            x['choices']['text'] if dset_name == 'arc' else
            [x['option1'], x['option2']] if dset_name == 'winogrande' else
            x['choices'] if dset_name == 'mmlu' else
            x['mc1_targets']['choices'] if dset_name == 'truthfulqa' else None
        )

    def make_index(self, answer, offset=0):
        answer = str(answer)
        if answer in ascii_uppercase:
            return ascii_uppercase.index(answer)
        if answer in [str(n) for n in range(10)]:
            return int(answer) - offset
        raise Exception(f"Unknown answer format: {answer}")

    def run_test(self, start_q, end_q):
        assert start_q < end_q
        num_prompts = end_q - start_q
        prompts, choices, q_strings, correct_ans = [], [], [], []
        for i in range(start_q, end_q):
            q_str, ch, corr_letter, _ = self.make_question(i)
            prompt = self.make_prompt(q_str, i, ch)
            prompts.append(prompt)
            choices.append(ch)
            q_strings.append(q_str)
            correct_ans.append(corr_letter)

        text_output, token_output, scores = self.model.generate(prompts)
        conf_prob, pred_letters1 = self.model.compute_confidence_levels(text_output, token_output, scores, choices, normalize=True)
        conf_logit, pred_letters2 = self.model.compute_confidence_levels(text_output, token_output, scores, choices, normalize=False)

        if pred_letters1 != pred_letters2:
            print("Warning: Predicted letters with normalized and unnormalized confidence levels do not match. It's probably because the values are super close, which is fine")
            for i in range(len(pred_letters1)):
                if pred_letters1[i] != pred_letters2[i]:
                    print(f"Prompt {i+1}: normalized: {pred_letters1[i]}, unnormalized: {pred_letters2[i]}")
        grades = []
        for i, (llm_output, pred) in enumerate(zip(text_output, pred_letters1)):
            print(f"Question {i + 1 + start_q}: {q_strings[i]}")
            print(f"LLM output: {llm_output}")
            correct = pred == correct_ans[i]
            grades.append(1 if correct else -1)
            print(f"LLM answer: {pred}. ({'correct' if correct else 'incorrect ' + correct_ans[i]})\n")
            conf_to_str = lambda x: 0 if t_to_str(x) == '' else t_to_str(x)
            print(f"Answer probabilities: {conf_prob[i]}")
            print(f"Answer logits: {conf_logit[i]}\n")

        return grades, conf_prob, conf_logit

    def get_output_filepath(self, conf_type):
        dataset_str = self.args['dataset'].split("/")[-1]
        prompt_name = {0: "first", 1: "second"}[self.args['prompt_phrasing']]
        few_shot_str = '' if self.args['few_shot_number'] == 0 else f"_few_shot_{self.args['few_shot_number']}"
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        return f"{out_dir}/d={dataset_str}_m={self.args['model']}_q={self.start_q}-{self.end_q}_c={conf_type}_p={prompt_name}_prompt{few_shot_str}.txt"

    def write_output(self, grades, conf_prob, conf_logit):
        groups = [('prob', conf_prob), ('logit', conf_logit)] if 'gpt' not in self.args['model'] else [('prob', conf_prob)] # we only have probabilities for the OpenAI models
        for group, conf in groups:
            path = self.get_output_filepath(group)
            print("\nWriting results to", path)
            with open(path, 'w') as f:
                f.write("grade confidence_levels\n")
                for g, c in zip(grades, conf):
                    conf_str = ','.join([str(x) for x in c])
                    g_str = "Correct" if g == 1 else "Wrong"
                    f.write(f"{g_str} {conf_str}\n")

    def make_question(self, i):
        data = self.questions[i]
        ch = self.get_choices(data)
        if len(ch) > 25:
            raise Exception("We only have 26 capital letters, so you can't have more than 26 answer options. Also why do you need that many?")
        q = self.get_q(data)
        correct_text = ch[self.get_a(data)]
        random.shuffle(ch)
        corr_letter = ascii_uppercase[ch.index(correct_text)]
        formatted = [f"{ascii_uppercase[idx]}. {opt}" for idx, opt in enumerate(ch)]
        return q + '\n' + '\n'.join(formatted), ch, corr_letter, correct_text

    def make_prompt(self, question_string, i, choices):
        response_marker = "Response:" if self.args['prompt_phrasing'] == 0 else "Answer:"
        # Add few shot examples if requested
        if self.args['few_shot_number'] > 0:
            valid_example_indices = [j for j in range(len(self.questions)) if j != i]
            example_indices = random.sample(valid_example_indices, self.args['few_shot_number'])
            example_questions = [self.make_question(j) for j in example_indices]
            q_number_str = lambda x: f' {x+1}' if self.args['few_shot_number'] > 1 else ''
            few_shot_qs = ''.join([f"Example question{q_number_str(j)}:\n{q}\n{response_marker} {a_letter}\n" for (j, (q,_,a_letter,a_text)) in enumerate(example_questions)])
            prefix = 'First, here are some example questions and the corresponding correct responses.' if self.args['few_shot_number'] > 1 else 'First, here is an example question and the corresponding correct response.'
            few_shot_string = '\n' + prefix + '\n\n' + few_shot_qs + 'Now for the actual question:'
        else:
            few_shot_string = '\nQuestion:' if self.args['prompt_phrasing'] == 0 else 'Now here is the question:'
        # Options for responses
        if len(choices) == 2:
            option_string = "A or B"
        else:
            option_string = ', '.join([ascii_uppercase[i] for i in range(len(choices)-1)]) + ', or ' + ascii_uppercase[len(choices)-1]
        # Now make the actual prompt
        if self.args['prompt_phrasing'] == 0:
            prompt = f"""Below is a multiple-choice question. Choose the letter that best answers the question. Keep your response brief and simply state the letter corresponding to your answer: {option_string}. {few_shot_string}
{question_string}
{response_marker} """
            # For some reason the final newline makes Falcon-7b act really weird
            return prompt if self.args['model'] != 'Falcon-7b' else prompt[:-1]
        elif self.args['prompt_phrasing'] == 1:
            return f"""You will be asked a multiple-choice question. Respond with the letter that corresponds to the correct answer: {option_string}. There is no need to provide an explanation, so your response should consist of just one letter. {few_shot_string}
{question_string}
{response_marker} """
        else:
            raise Exception(f"Unknown phrasing option: {self.args['prompt_phrasing']}. Must be 0 or 1.")    

def main():
    random.seed(2549900867) # We'll randomize the order of questions and of answer choices, but we want every run to have the same randomization
    args = parse_args()
    test = Test(args)

    # Exit if all of the results files already exist.
    conf_types = ['prob', 'logit'] if 'gpt' not in args['model'] else ['prob']
    output_filepaths = [test.get_output_filepath(conf_type) for conf_type in conf_types]
    if all([os.path.exists(output_filepath) for output_filepath in output_filepaths]):
        print(f"Results files {output_filepaths} already exist. Exiting.")
        return
    
    all_grades, all_conf_prob, all_conf_logit = [], [], []
    for start_q in range(test.start_q, test.end_q, args['batch_size']):
        end_q = min(start_q + args['batch_size'], test.end_q)
        if args['batch_size'] > 1:
            print(f"\nSTARTING NEW BATCH: questions {start_q} to {end_q}\n")
        (grades, conf_prob, conf_logit) = test.run_test(start_q, end_q)
        all_grades += grades
        all_conf_prob += conf_prob
        all_conf_logit += conf_logit
        
    if len(all_grades) > 0: # E.g. if the dataset only has 817 qs but you ask to run qs 1000-1500
        test.write_output(all_grades, all_conf_prob, all_conf_logit)
    else:
        print("The question range you provided is empty. This could either be because endq < startq or because the dataset is too small.")

if __name__ == '__main__':
    main()
