import json
import sys

from orca.dataset_readers.bidaf_baseline import BiDAFBaselineReader
from orca.dataset_readers.bidaf_copynet import BiDAFCopyNetDatasetReader
from orca.dataset_readers.copynet_baseline import CopyNetBaselineDatasetReader
from orca.models.bidaf_copynet import BiDAFCopyNetSeq2Seq
from orca.predictors.sharc_predictor import ShARCPredictor

from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from nltk.translate.bleu_score import sentence_bleu

archive = load_archive(sys.argv[1])
predictor = Predictor.from_archive(archive, 'sharc_predictor')

bleu_score_sum = [0, 0, 0, 0, 0]
count = 0

tokenizer = WordTokenizer()
def tokenize(text):
    tokens_list = tokenizer.tokenize(text)
    return [token.text for token in tokens_list]

def history_to_string(history):
    output = ''
    first_qa = True
    for qa in history:
        if not first_qa:
            output += '\n'
        output += 'Q: ' + qa['follow_up_question'] + '\n'
        output += 'A: ' + qa['follow_up_answer']
        first_qa = False
    return output

buffer = ''
with open('./sharc1-official/json/sharc_dev.json', 'r') as dev_file:
    dev_json = json.load(dev_file)
    for utterance in dev_json:
        answer = utterance['answer']
        scenario = utterance['scenario']
        if answer in ['Yes', 'No', 'Irrelevant'] or scenario is not '':
            continue

        result = predictor.predict_json(utterance)
        if 'best_span_str' in result.keys(): # BiDAF
            predicted_answer = result['best_span_str']
        elif 'prediction' in result.keys(): # CopyNet
            predicted_answer = ' '.join(result['prediction'])
        elif 'predicted_tokens' in result.keys(): # CopyNet Baseline
            predicted_answer = ' '.join(result['predicted_tokens'][0])
        else:
            raise ValueError('Error')

        buffer += 'RULE TEXT: ' + utterance['snippet'] + '\n'
        buffer += 'SCENARIO: ' + utterance['scenario'] + '\n'          
        buffer += 'QUESTION: ' + utterance['question'] + '\n'
        buffer += 'HISTORY: ' + history_to_string(utterance['history']) + '\n'
        buffer += 'GOLD ANSWER: ' + answer + '\n'
        buffer += 'PREDICTED ANSWER: ' + str(predicted_answer) + '\n\n\n'

        bleu_score_sum[1] += sentence_bleu([tokenize(predicted_answer)], tokenize(answer), weights=(1,))
        bleu_score_sum[2] += sentence_bleu([tokenize(predicted_answer)], tokenize(answer), weights=(0.5, 0.5))
        bleu_score_sum[3] += sentence_bleu([tokenize(predicted_answer)], tokenize(answer), weights=(1/3, 1/3, 1/3))
        bleu_score_sum[4] += sentence_bleu([tokenize(predicted_answer)], tokenize(answer), weights=(1/4, 1/4, 1/4, 1/4))

        count += 1

with open('evaluation', 'w', encoding='utf-8') as out_file:
    for i in [1, 2, 3, 4]:
        print("Bleu-{} score: {}".format(i, bleu_score_sum[i]/count))
        out_file.write("Bleu-{} score: {}".format(i, bleu_score_sum[i]/count) + '\n')
    out_file.write('\n\n')
    out_file.write(buffer)