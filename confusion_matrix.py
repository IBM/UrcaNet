"""
python confusion_matrix.py bcft3_pred bcft_3 gold ./bcft3_ea/
"""

import json
import os
import sys

import pandas as pd

from evaluate import prettify_utterance

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python confusion_matrix.py <prediction_file> <gold_file> <output_folder>")
        sys.exit()
    prediction_file = sys.argv[1]
    gold_file = sys.argv[2]
    output_folder = sys.argv[3]

    with open(gold_file) as file:
        gold_answers_json = json.load(file)
    with open(prediction_file) as file:
        predicted_answers_json = json.load(file)

    combined_answers = {}
    for gold_answer, predicted_answer in zip(gold_answers_json, predicted_answers_json):
        utterance_id = gold_answer['utterance_id']
        assert utterance_id == predicted_answer['utterance_id']
        combined_answers[utterance_id] = (gold_answer['answer'], predicted_answer['answer'])

    confusion_matrix_samples = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append([])
        confusion_matrix_samples.append(row)

    action_map = {'Irrelevant': 0, 'More': 1, 'No': 2, 'Yes': 3}
    action_map_rev = {0: 'irrl', 1: 'more', 2: 'no', 3: 'yes'}

    for uid, (gold_answer, predicted_answer) in combined_answers.items():
        gold_action = gold_answer if gold_answer in ['Yes', 'No', 'Irrelevant'] else 'More'
        prediction_action = predicted_answer if predicted_answer in ['Yes', 'No', 'Irrelevant'] else 'More'
        confusion_matrix_samples[action_map[gold_action]][action_map[prediction_action]].append((uid, gold_answer, predicted_answer))

    dev_dataset = 'sharc1-official/json/sharc_dev.json'
    with open(dev_dataset) as file:
            dev_dataset_json = json.load(file)
    dev_dataset_dict = {utterance['utterance_id']: utterance for utterance in dev_dataset_json}

    os.makedirs(output_folder, exist_ok=True)

    confusion_matrix = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(len(confusion_matrix_samples[i][j]))
        confusion_matrix.append(row)
    labels = ['Irrl', 'More', 'No', 'Yes']
    cm_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    print(cm_df)
    with open(os.path.join(output_folder, 'confusion_matrix'), 'w') as file:
        file.write(str(cm_df))

    for i in range(4):
        for j in range(4):
            filename = 'gold_{}_pred_{}'.format(action_map_rev[i], action_map_rev[j])
            filename = os.path.join(output_folder, filename)
            with open(filename, 'w') as file:
                file.write('')
            for uid, _, predicted_answer in confusion_matrix_samples[i][j]:
                with open(filename, 'a') as file:
                    file.write(prettify_utterance(dev_dataset_dict[uid], predicted_answer) + '\n\n')