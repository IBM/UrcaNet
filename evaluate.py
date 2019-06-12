"""
Usage: python evaluate.py <archived_model> <summary_file_name>
"""


from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import json
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
import sys

import evaluator
from orca.dataset_readers.bidaf_baseline import BiDAFBaselineReader
from orca.dataset_readers.bidaf_copynet import BiDAFCopyNetDatasetReader
from orca.dataset_readers.bidaf_copynet_pipeline import BiDAFCopyNetPipelineDatasetReader
from orca.dataset_readers.copynet_baseline import CopyNetBaselineDatasetReader
from orca.dataset_readers.bidaf_baseline_ft import BiDAFBaselineFTReader
from orca.dataset_readers.bidaf_copynet_ft import BiDAFCopyNetFTDatasetReader
from orca.dataset_readers.sharc_net import ShARCNetDatasetReader
from orca.models.bidaf_modified import BidirectionalAttentionFlowModified
from orca.models.bidaf_copynet import BiDAFCopyNetSeq2Seq
from orca.models.bidaf_ft import BidirectionalAttentionFlowFT
from orca.models.bidaf_copynet_ft import BiDAFCopyNetFTSeq2Seq
from orca.models.sharc_net import ShARCNet
from orca.predictors.sharc_predictor import ShARCPredictor

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

def get_batch_prediction(predictor, utterances):
    model_outputs = predictor.predict_batch_json(utterances)

    predicted_answers = []
    for model_output in model_outputs:
        predicted_action = model_output.get('label', None)
        
        if 'best_span_str' in model_output.keys(): # BiDAF
            predicted_answer = model_output['best_span_str']
        elif 'prediction' in model_output.keys(): # CopyNet
            predicted_answer = ' '.join(model_output['prediction'])
        elif 'predicted_tokens' in model_output.keys(): # CopyNet Baseline
            predicted_answer = ' '.join(model_output['predicted_tokens'][0])
        else:
            raise ValueError('Can\'t get prediction.')

        if predicted_action and predicted_action != 'More':
            predicted_answer = predicted_action
        predicted_answers.append(predicted_answer)

    return predicted_answers

def prettify_utterance(utterance, predicted_answer):
    output = 'RULE TEXT: ' + utterance['snippet'] + '\n'
    output += 'SCENARIO: ' + utterance['scenario'] + '\n'          
    output += 'QUESTION: ' + utterance['question'] + '\n'
    output += 'HISTORY: ' + history_to_string(utterance['history']) + '\n'
    output += 'GOLD ANSWER: ' + utterance['answer'] + '\n'
    output += 'PREDICTED ANSWER: ' + str(predicted_answer)
    return output

def make_json(answers, filename):
    output_json = []
    for utterance_id, answer in answers:
        output_json.append({'utterance_id': utterance_id, 'answer': answer})
    with open(filename, 'w') as file:
        json.dump(output_json, file)

def prettify_dict(Dict):
    output = ''
    for key, value in Dict.items():
        output += '{}: {}\n'.format(key, value)
    return output

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == "__main__":
    task = 'full' # 'qgen'

    if len(sys.argv) != 3:
        print('Usage: python evaluate.py <archived_model> <summary_file_name>')
        sys.exit()
    archived_model = sys.argv[1]
    summary_file = sys.argv[2]

    archive = load_archive(archived_model, cuda_device=0)
    predictor = Predictor.from_archive(archive, 'sharc_predictor')

    predicted_answers = []
    gold_answers = []
    summary = ''
    mode_map = {'full': 'combined', 'qgen': 'follow_ups'}
    dev_dataset = './sharc1-official/json/sharc_dev.json'

    with open(dev_dataset, 'r') as dev_file:
        dev_json = json.load(dev_file)

    predicted_answers_ = []
    for utterances in tqdm(chunks(dev_json, 40)):
        predicted_answers_ += get_batch_prediction(predictor, utterances)

    for utterance, predicted_answer in zip(dev_json, predicted_answers_):
        answer = utterance['answer']
        scenario = utterance['scenario']
        if task == 'qgen' and (answer in ['Yes', 'No', 'Irrelevant'] or scenario != ''):
            continue
        summary += prettify_utterance(utterance, predicted_answer) + '\n\n\n'
        gold_answers.append((utterance['utterance_id'], answer))
        predicted_answers.append((utterance['utterance_id'], predicted_answer))

    make_json(gold_answers, summary_file + '_gold')
    make_json(predicted_answers, summary_file + '_prediction')
    results = evaluator.evaluate(summary_file + '_gold', summary_file + '_prediction', mode=mode_map[task])
    print(prettify_dict(results))
    summary = prettify_dict(results) + '\n\n' + summary 

    with open(summary_file, 'w', encoding='utf-8') as file:
        file.write(summary)