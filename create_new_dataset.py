import copy
from collections import defaultdict
from hashlib import sha1
import itertools
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split

random.seed(102)

def class_distribution(utterances):
    class_counts = {'Irrelevant': 0, 'More':0, 'Yes': 0, 'No': 0}
    for utterance in utterances:
        answer = utterance['answer']
        class_ = answer if answer in ['Yes', 'No', 'Irrelevant'] else 'More'
        class_counts[class_] += 1
    return class_counts

def build_scenario_evidence_map(utterances):
    """Builds a map from scenario to evidence"""
    scenario_evidence_map = dict()
    for utterance in utterances:
        scenario = utterance['scenario']
        evidence = utterance['evidence']
        scenario_evidence_map[scenario] = evidence
    return scenario_evidence_map

def build_question_scanerio_map(utterances):
    """Builds a map from question to scenarios"""
    question_scenario_map = dict()
    for utterance in utterances:
        question = utterance['question']
        scenario = utterance['scenario']
        if scenario == '':
            continue
        if question not in question_scenario_map.keys():
            question_scenario_map[question] = set([scenario])
        else:
            question_scenario_map[question].add(scenario)
    return question_scenario_map

def add_scenarios(utterances, question_scenario_map, scenario_evidence_map):
    """Adds relevant scenario when gold label is 'Irrelevant'"""
    augmented_utterances = []
    for utterance in utterances:
        answer = utterance['answer']
        question = utterance['question']
        scenario = utterance['scenario']
        history = utterance['history']
        del utterance['utterance_id'] 
        utterance['utterance_id'] = sha1(str(utterance).encode('utf-8')).hexdigest()
        augmented_utterances.append(utterance)
        if answer == 'Irrelevant' and scenario == '':
            relevant_scenarios = list(question_scenario_map.get(question, []))
            random.shuffle(relevant_scenarios)
            for scenario in relevant_scenarios[:12]:
                new_utterance = copy.deepcopy(utterance)
                new_utterance['scenario'] = scenario
                new_utterance['evidence'] = scenario_evidence_map[scenario]
                del new_utterance['utterance_id'] 
                new_utterance['utterance_id'] = sha1(str(new_utterance).encode('utf-8')).hexdigest()
                augmented_utterances.append(new_utterance)     
    return augmented_utterances

def shuffle_history(utterances):
    """Shuffles history if present."""
    augmented_utterances = []
    for utterance in utterances:        
        history = utterance['history']
        permutations = list(itertools.permutations(history))
        random.shuffle(permutations)
        for history in permutations[:2]:
            new_utterance = copy.deepcopy(utterance)
            new_utterance['history'] = list(history)
            del new_utterance['utterance_id'] 
            new_utterance['utterance_id'] = sha1(str(new_utterance).encode('utf-8')).hexdigest()
            augmented_utterances.append(new_utterance)
    return augmented_utterances

def clean_dataset(dataset):
    fixed_dataset = copy.deepcopy(dataset)
    # correct spelling errors 
    for utterance in fixed_dataset:
        for followup_qa in utterance['history'] + utterance['evidence']:
            if 'followup_question' in followup_qa:
                followup_qa['follow_up_question'] = followup_qa.pop('followup_question')
            if 'followup_answer' in followup_qa:
                followup_qa['follow_up_answer'] = followup_qa.pop('followup_answer')
    
    # remove duplicates (approx)
    cleaned_dataset = []
    hashes = set()
    for utterance in fixed_dataset:
        utterance_id = utterance.pop('utterance_id')
        hash_ = sha1(str(utterance).encode('utf-8')).hexdigest()
        if hash_ in hashes:
            continue
        else:
            hashes.add(hash_)
            utterance['utterance_id'] = utterance_id
            cleaned_dataset.append(utterance)
    return cleaned_dataset

def data_characteristics(filename):
    characteristics = {}
    with open(filename, 'r') as file:
        dataset = json.load(file)
    
    dataset_size = len(dataset)
    turn_length_counts = np.zeros(7)
    class_map = {'Irrelevant': 0, 'More': 1, 'No': 2, 'Yes': 3}
    class_counts = np.zeros(4) 
    star_count = 0
    scenario_present = 0
    for utterance in dataset:
        answer = utterance['answer']
        history = utterance['history']
        rule = utterance['snippet']
        scenario = utterance['scenario']
        turn_length = min(len(history), 6)
        turn_length_counts[turn_length] += 1
        class_ = answer if answer in ['Yes', 'No', 'Irrelevant'] else 'More'
        class_counts[class_map[class_]] += 1
        if '*' in rule:
            star_count += 1
        if scenario != '':
            scenario_present += 1

    characteristics['dataset_size'] = dataset_size
    characteristics['class_distribution']  = np.around(class_counts / class_counts.sum(), 2)
    characteristics['turn_length_distribution'] = np.around(turn_length_counts / turn_length_counts.sum(), 2)
    characteristics['star_present'] = round(star_count / dataset_size, 2)
    characteristics['scenario_present'] = round(scenario_present / dataset_size, 2)
    return characteristics


if __name__ == "__main__":
    train_dataset = 'sharc1-official/json/sharc_train.json'
    dev_dataset = 'sharc1-official/json/sharc_dev.json'
    with open(train_dataset) as file:
            train_json = json.load(file)
    with open(dev_dataset) as file:
            dev_json = json.load(file)
    dataset_json = train_json + dev_json
    dataset_json = clean_dataset(dataset_json)
    dataset_json = shuffle_history(dataset_json)
    question_scenario_map = build_question_scanerio_map(dataset_json)
    scenario_evidence_map = build_scenario_evidence_map(dataset_json)
    dataset_json = add_scenarios(dataset_json, question_scenario_map, scenario_evidence_map)

    count = 0
    total = 0
    for utterance in dataset_json:
        answer = utterance['answer']
        class_ = answer if answer in ['Yes', 'No', 'Irrelevant'] else 'More'
        if class_ == 'Irrelevant':
            total += 1
            if not utterance['scenario']:
                count += 1

    # We do this so that there are no utterances in train and dev with same tree_id
    data_dict = defaultdict(list)
    for utterance in dataset_json:
        data_dict[utterance['tree_id']].append(utterance)

    data_items = list(data_dict.items())
    train_items, other_items = train_test_split(data_items, test_size=0.2, random_state=132)
    dev_items, val_items = train_test_split(other_items, test_size=0.5, random_state=9)

    train_data = []
    for _, utterances in train_items:
        train_data += utterances
    val_data = []
    for _, utterances in val_items:
        val_data += utterances
    dev_data = []
    for _, utterances in dev_items:
        dev_data += utterances

    train_file = './sharc1-official/json/sharc_new_train.json'
    val_file = './sharc1-official/json/sharc_new_val.json'
    dev_file = './sharc1-official/json/sharc_new_dev.json'

    with open(train_file, 'w') as file:
        json.dump(train_data, file)
    with open(val_file, 'w') as file:
        json.dump(val_data, file)
    with open(dev_file, 'w') as file:
        json.dump(dev_data, file)

    print(data_characteristics(train_file))
    print(data_characteristics(val_file))
    print(data_characteristics(dev_file))