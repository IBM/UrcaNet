import argparse
from difflib import SequenceMatcher
import json
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm

from allennlp.data.tokenizers import WordTokenizer
from orca.modules.word_splitter import SpacyWordSplitterModified

tokenizer = WordTokenizer(word_splitter = SpacyWordSplitterModified(never_split=["[SEP]"]))
wp_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').wordpiece_tokenizer

def get_number_word_pieces(text):
    num_word_pieces = 0
    for token in tokenizer.tokenize(text): 
        num_word_pieces += len(wp_tokenizer.tokenize(token.text))
    return num_word_pieces

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='path to input file in QuAC format')
    parser.add_argument('output_file', help='path where to save file in ShARC format')
    parser.add_argument('--max_word_pieces', type=int, default='3000', 
                        help='max word pieces (including both rule and question) to keep')
    parser.add_argument("--debug", help="save only few examples", action="store_true")

    args = parser.parse_args()

    with open(args.input_file) as file:
        dataset = json.load(file)

    utterances = []
    skipped = 0
    total = 0
    for file_ in tqdm(dataset['data']):
        rule = file_['story']
        rule_wps = get_number_word_pieces(rule)
        history = []
        for question, answer in zip(file_['questions'], file_['answers']):            
            question = question['input_text']
            answer = answer['span_text']
            
            question_wps = get_number_word_pieces(question)
            total += 1
            if rule_wps + question_wps >= args.max_word_pieces:
                skipped += 1
                continue

            utterances.append({'snippet': rule,
                            'question': question,
                            'scenario': '',
                            'history': history,
                            'evidence': [],
                            'answer': answer})
            followup_qa = {}
            followup_qa['follow_up_question'] = answer
            followup_qa['follow_up_answer'] = question
            history.append(followup_qa)
        if args.debug:
            break
    print(f'Skipped {skipped * 100 / total:.2f}% of instances.')

    with open(args.output_file, 'w') as file:
        json.dump(utterances, file)

