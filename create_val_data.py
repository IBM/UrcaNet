from collections import defaultdict
import json
import random
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train_file = './sharc1-official/json/sharc_train.json'
    with open(train_file, 'r') as file:
        train_json = json.load(file)

    # We do this so that there are no utterances in train and dev with same tree_id
    train_dict = defaultdict(list)
    for utterance in train_json:
        train_dict[utterance['tree_id']].append(utterance)
    
    train_items, val_items = train_test_split(list(train_dict.items()), test_size=0.1)

    train_data = []
    for _, utterances in train_items:
        train_data += utterances
    val_data = []
    for _, utterances in val_items:
        val_data += utterances
    
    train_split_file = './sharc1-official/json/sharc_train_split.json'
    val_split_file = './sharc1-official/json/sharc_val_split.json'
    with open(train_split_file, 'w') as file:
        json.dump(train_data, file)
    with open(val_split_file, 'w') as file:
        json.dump(val_data, file)