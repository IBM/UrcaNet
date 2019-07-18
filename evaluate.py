"""
Usage: python evaluate.py <archived_model> <summary_file_name>
"""

from allennlp.common import Params
from allennlp.data import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import argparse
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import torch
from tqdm import tqdm
import sys

import evaluator
from orca.dataset_readers.bidaf_baseline import BiDAFBaselineReader
from orca.dataset_readers.bidaf_copynet import BiDAFCopyNetDatasetReader
from orca.dataset_readers.bidaf_copynet_pipeline import BiDAFCopyNetPipelineDatasetReader
from orca.dataset_readers.copynet_baseline import CopyNetBaselineDatasetReader
from orca.dataset_readers.copynet_pipeline import CopyNetPipelineDatasetReader
from orca.dataset_readers.bidaf_baseline_ft import BiDAFBaselineFTReader
from orca.dataset_readers.bidaf_copynet_ft import BiDAFCopyNetFTDatasetReader
from orca.dataset_readers.sharc_net import ShARCNetDatasetReader
from orca.dataset_readers.bert_qa import BertQAReader
from orca.dataset_readers.bert_copynet import BertCopyNetDatasetReader
from orca.dataset_readers.bert_copynet_dual import BertCopyNetDualDatasetReader
from orca.models.bidaf_modified import BidirectionalAttentionFlowModified
from orca.models.bidaf_copynet import BiDAFCopyNetSeq2Seq
from orca.models.bidaf_ft import BidirectionalAttentionFlowFT
from orca.models.bidaf_copynet_ft import BiDAFCopyNetFTSeq2Seq
from orca.models.bert_qa import BertQA
from orca.models.bert_copynet import BertCopyNetFTSeq2Seq
from orca.models.bert_copynet_dual import BertCopyNetDualSeq2Seq
from orca.models.copynet_pipeline import CopyNetPipeline
from orca.models.sharc_net import ShARCNet
from orca.predictors.sharc_predictor import ShARCPredictor
from orca.modules.bert_token_embedder import PretrainedBertModifiedEmbedder
from orca.modules.bert_indexer import PretrainedBertHistoryAugmentedIndexer
from orca.modules.word_splitter import SpacyWordSplitterModified

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

def get_batch_prediction(predictor, utterances, perfect_classification=False):
    model_outputs = predictor.predict_batch_json(utterances)

    predicted_answers = []
    for utterance, model_output in zip(utterances, model_outputs):
        if perfect_classification:
            predicted_action = utterance['answer'] if utterance['answer'] in ['Yes', 'No', 'Irrelevant'] else 'More'
        else:
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

def get_best_predictor(model_dir):
    params = Params.from_file(os.path.join(model_dir, "config.json"))
    vocab = Vocabulary.from_files(os.path.join(model_dir, "vocabulary"))

    config = params.duplicate()
    model = Model.from_params(vocab=vocab, params=config.pop('model'))
    map_location = None if torch.cuda.is_available() else 'cpu'
    with open(os.path.join(model_dir, "best.th"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    config = params.duplicate()    
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    predictor = Predictor.by_name('sharc_predictor')(model, dataset_reader)
    return predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archived_model", help="path to folder containing the archived model")
    parser.add_argument("summary_file", help="filename for the summary generated")
    parser.add_argument("--test_dataset", help="path to test dataset", default='./sharc1-official/json/sharc_dev.json')
    parser.add_argument("--task", help="task to evaluate model on", default='full', choices = ['full', 'qgen'])
    parser.add_argument("--bleu_pc", help="calculate bleu as if classification is perfect", action="store_true")
    args = parser.parse_args()
    
    task = args.task
    archived_model = args.archived_model
    summary_file = args.summary_file
    dev_dataset = args.test_dataset

    try:
        cuda_device = 0 if torch.cuda.is_available() else -1
        archive = load_archive(archived_model, cuda_device=cuda_device)
        predictor = Predictor.from_archive(archive, 'sharc_predictor')
    except FileNotFoundError:
        print('Model still training. Using best weights..')
        predictor = get_best_predictor(archived_model) # Assuming archived_model is folder

    predicted_answers = []
    gold_answers = []
    summary = ''
    mode_map = {'full': 'combined', 'qgen': 'follow_ups'}

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

    if args.bleu_pc:
        predicted_answers_pc = []
        predicted_answers_pc_ = []
        for utterances in tqdm(chunks(dev_json, 40)):
            predicted_answers_pc_ += get_batch_prediction(predictor, utterances, perfect_classification=True)
        for utterance, predicted_answer in zip(dev_json, predicted_answers_pc_):
            answer = utterance['answer']
            scenario = utterance['scenario']
            if task == 'qgen' and (answer in ['Yes', 'No', 'Irrelevant'] or scenario != ''):
                continue
            predicted_answers_pc.append((utterance['utterance_id'], predicted_answer))

    make_json(gold_answers, summary_file + '_gold')
    make_json(predicted_answers, summary_file + '_prediction')
    results = evaluator.evaluate(summary_file + '_gold', summary_file + '_prediction', mode=mode_map[task])
    print(prettify_dict(results))
    if args.bleu_pc:
        make_json(predicted_answers_pc, summary_file + '_prediction_pc')
        results_pc = evaluator.evaluate(summary_file + '_gold', summary_file + '_prediction_pc', mode=mode_map[task])
        print(prettify_dict(results_pc))
    summary = prettify_dict(results) + '\n\n' + summary 

    with open(summary_file, 'w', encoding='utf-8') as file:
        file.write(summary)