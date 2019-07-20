import torch
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertEncoder, BertPooler, BertLayerNorm, BertPreTrainedModel

class BertEmbeddingsModified(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddingsModified, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.history_embeddings = nn.Embedding(4, config.hidden_size, padding_idx=0)
        self.turn_embeddings = nn.Embedding(8, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, history_encoding=None, turn_encoding=None, scenario_encoding=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if history_encoding is None:
            history_encoding = torch.zeros_like(input_ids)
        if turn_encoding is None:
            turn_encoding = torch.zeros_like(input_ids)
        if scenario_encoding is None:
            scenario_encoding = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        history_embeddings = self.history_embeddings(history_encoding)
        scenario_embeddings = self.history_embeddings(scenario_encoding)
        turn_embeddings = self.turn_embeddings(turn_encoding)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings + history_embeddings + turn_embeddings + scenario_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelModified(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelModified, self).__init__(config)
        self.embeddings = BertEmbeddingsModified(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, history_encoding=None, turn_encoding=None, scenario_encoding=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if history_encoding is None:
            history_encoding = torch.zeros_like(input_ids)
        if turn_encoding is None:
            turn_encoding = torch.zeros_like(input_ids)
        if scenario_encoding is None:
            scenario_encoding = torch.zeros_like(input_ids)
            
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, history_encoding, turn_encoding, scenario_encoding)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output