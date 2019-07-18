{
  "dataset_reader": {
    "type": "copynet_pipeline",
    "add_rule": true,
    "embed_span": true,
    "add_question": true,
    "add_followup_ques": true,
    "target_namespace": "target_tokens",
    "span_predictor_model": "/dccstor/sharc/models/bert_qa_base_afwhs",
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
    "type": "copynet_pipeline",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "pretrained_file": "http://sfs.uni-tuebingen.de/~ccoltekin/courses/ml/data/glove.6B.200d.txt.gz",
          "embedding_dim": 200,
          "trainable": true
        },
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 200,
      "bidirectional": true,
      "hidden_size": 200,
      "num_layers": 1
    },
    "attention": {
      "type": "linear",
      "tensor_1_dim": 400,
      "tensor_2_dim": 400
    },
    "target_embedding_dim": 200,
    "beam_size": 5,
    "max_decoding_steps": 50
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 32,
    "sorting_keys": [["source_tokens", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": "+BLEU",
    "optimizer": {
      "type": "adam",
    }
  }
}