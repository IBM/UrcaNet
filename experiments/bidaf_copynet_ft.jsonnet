{
  "dataset_reader": {
    "target_namespace": "target_tokens",
    "type": "bidaf_copynet_ft",
    "bidaf_input_tokenizer": {
      "type": "word",
    },
    "source_tokenizer": {
      "type": "word",
    },
    "target_tokenizer": {
      "type": "word"
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
    "type": "bidaf_copynet_ft",
    "pretrained_bidaf": false,
    "bidaf_model": {
      "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                "embedding_dim": 100,
                "trainable": true
            }
        }
      },
      "num_highway_layers": 2,
      "phrase_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 100,
        "hidden_size": 100,
        "num_layers": 1
      },
      "similarity_function": {
        "type": "linear",
        "combination": "x,y,x*y",
        "tensor_1_dim": 200,
        "tensor_2_dim": 200
      },
      "modeling_layer": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 800,
        "hidden_size": 100,
        "num_layers": 2,
        "dropout": 0.2
      },
      "span_end_encoder": {
        "type": "lstm",
        "bidirectional": true,
        "input_size": 1400,
        "hidden_size": 100,
        "num_layers": 1
      },
      "dropout": 0.2
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
      "hidden_size": 200,
      "num_layers": 1
    },
    "attention": {
      "type": "bilinear",
      "vector_dim": 200,
      "matrix_dim": 200
    },
    "target_embedding_dim": 100,
    "beam_size": 3,
    "max_decoding_steps": 50,
    "dropout": 0.2
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 40,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 10,
    "validation_metric": "+agg_metric",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "weight_decay": 1e-3,
      "lr": 1e-4
    }
  }
}