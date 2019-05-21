import json

from orca.dataset_readers.bidaf_baseline import ShARCBiDAFReader
from orca.predictors.bidaf_baseline import BidafPredictor

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from nltk.translate.bleu_score import sentence_bleu

archive = load_archive('./temp/sharc_bidaf/model.tar.gz')
predictor = Predictor.from_archive(archive, 'bidaf_baseline')

total_bleu1_score = 0
total_bleu2_score = 0
total_bleu3_score = 0
total_bleu4_score = 0
count = 0
with open('./sharc1-official/json/sharc_dev.json', 'r') as dev_file:
    dev_json = json.load(dev_file)
    for utterance in dev_json:
        answer = utterance['answer']
        if answer in ['Yes', 'No', 'Irrelevant']:
            continue

        result = predictor.predict_json(utterance)
        predicted_answer = result['best_span_str']
        total_bleu1_score += sentence_bleu([predicted_answer.split()], answer.split(), weights=(1,))
        total_bleu2_score += sentence_bleu([predicted_answer.split()], answer.split(), weights=(0.5, 0.5))
        total_bleu3_score += sentence_bleu([predicted_answer.split()], answer.split(), weights=(1/3, 1/3, 1/3))
        total_bleu4_score += sentence_bleu([predicted_answer.split()], answer.split(), weights=(1/4, 1/4, 1/4, 1/4))

        count += 1

print("Average Bleu-1 Score: {}".format(total_bleu1_score/count))
print("Average Bleu-2 Score: {}".format(total_bleu2_score/count))
print("Average Bleu-3 Score: {}".format(total_bleu3_score/count))
print("Average Bleu-4 Score: {}".format(total_bleu4_score/count))

