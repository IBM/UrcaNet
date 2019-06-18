// Configuration for the a machine comprehension model based on:
//   Seo, Min Joon et al. “Bidirectional Attention Flow for Machine Comprehension.”
//   ArXiv/1611.01603 (2016)
{
  "dataset_reader": {
    "type": "bert_qa",
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "spacy_modified",
        "never_split": ["[SEP]"], 
      }
    },
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": true,
      },
    },
  },
  "train_data_path": "sharc1-official/json/sharc_train_split.json",
  "validation_data_path": "sharc1-official/json/sharc_val_split.json",
  "model": {
    "type": "bert_qa",
    "text_field_embedder": {
      "type": "basic",
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained-modified",
          "pretrained_model": "bert-base-uncased",
          "requires_grad": true,
          "top_layer_only": true
        },
      },
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets", "bert-type-ids"]
      },
      "allow_unmatched_keys": true
    },
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["bert_input", "num_tokens"]],
    "batch_size": 12
  },

  "trainer": {
    "type": "modified_trainer",
    "minimal_save": true,
    "num_epochs": 50,
    "patience": 10,
    "validation_metric": "+agg_metric",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 1e-5,
    },
  }
}
