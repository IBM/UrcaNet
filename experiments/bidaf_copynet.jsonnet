{
  "dataset_reader": {
    "target_namespace": "target_tokens",
    "type": "bidaf_copynet",
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
  "train_data_path": "sharc1-official/json/sharc_train.json",
  "validation_data_path": "sharc1-official/json/sharc_dev.json",
  "model": {
    "type": "bidaf_copynet",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": false
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
    "beam_size": 5,
    "max_decoding_steps": 50
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 40,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
    }
  }
}