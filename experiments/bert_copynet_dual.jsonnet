{
  "dataset_reader": {
    "target_namespace": "target_tokens",
    "type": "bert_copynet_dual",
    "bert_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "spacy_modified",
        "never_split": ["[SEP]"], 
      }
    },
    "source_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "spacy_modified",
        "never_split": ["[SEP]"], 
      }    
    },
    "bert_token_indexers": {
      "bert": {
        "type": "bert-pretrained_hist_aug",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": true,
      },
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    }
  },
  "train_data_path": "sharc1-official/json/sharc_train_split.json",
  "validation_data_path": "sharc1-official/json/sharc_val_split.json",
  "model": {
    "type": "bert_copynet_dual",
    "bert_model1": {
        "text_field_embedder": {
        "type": "basic",
        "token_embedders": {
          "bert": {
            "type": "bert-pretrained-modified",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": true,
            "top_layer_only": false
          },
        },
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets", "bert-type-ids", "history_encoding"]
      },
        "allow_unmatched_keys": true
      }
    },
    "bert_model2": {
        "text_field_embedder": {
        "type": "basic",
        "token_embedders": {
          "bert": {
            "type": "bert-pretrained-modified",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": true,
            "top_layer_only": false
          },
        },
        "embedder_to_indexer_map": {
          "bert": ["bert", "bert-offsets", "bert-type-ids"]
        },
        "allow_unmatched_keys": true
      }
    },
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": true
        },
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 100,
      "hidden_size": 768,
      "num_layers": 1
    },
    "attention": {
      "type": "bilinear",
      "vector_dim": 768,
      "matrix_dim": 768
    },
    "target_embedding_dim": 768,
    "beam_size": 3,
    "max_decoding_steps": 50,
    "dropout": 0.2
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 6,
    "sorting_keys": [["bert_input2", "num_tokens"]]
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
      "lr": 1e-5
    },
  }
}