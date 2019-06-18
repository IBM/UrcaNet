import logging
from typing import Any, Dict, List, Optional

import numpy
import torch
from torch.nn.functional import cross_entropy, nll_loss, log_softmax

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from orca.modules.bert_token_embedder import PretrainedBertModifiedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("bert_qa")
class BertQA(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BertQA, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        embedding_dim = self._text_field_embedder.get_output_dim()
        self._action_predictor = torch.nn.Linear(embedding_dim, 4)
        self._span_predictor = torch.nn.Linear(embedding_dim, 2)
        self._action_accuracy = CategoricalAccuracy()
        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        initializer(self)

    def forward(self,  # type: ignore
                bert_input: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question tokens, passage tokens, original passage
            text, and token offsets into the passage for each instance in the batch.  The length
            of this list should be the batch size, and each dictionary should have the keys
            ``question_tokens``, ``passage_tokens``, ``original_passage``, and ``token_offsets``.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """

        # Shape: (batch_size, bert_input_len + 1, embedding_dim)
        bert_output = self._text_field_embedder(bert_input)
        # Shape: (batch_size, embedding_dim)
        pooled_output = bert_output[:, 0]
        # Shape: (batch_size, bert_input_len, embedding_dim)
        bert_output = bert_output[:, 1:, :] 
       
        if bert_input['bert-type-ids'].max() == 0:
            raise ValueError("Incorrect type-id.")

        # Shape: (batch_size, bert_input_len)
        input_type_ids = self.wordpiece_to_tokens(bert_input['bert-type-ids'], bert_input['bert-offsets']).float() 
        # Shape: (batch_size, bert_input_len)
        input_mask = util.get_text_field_mask(bert_input).float()
        passage_mask = input_mask - input_type_ids # works only with one [SEP]
        # Shape: (batch_size, bert_input_len, embedding_dim)
        passage_representation = bert_output * passage_mask.unsqueeze(2)
        # Shape: (batch_size, passage_len, embedding_dim)
        passage_representation = passage_representation[:, passage_mask.sum(dim=0) > 0, :]
        # Shape: (batch_size, passage_len)        
        passage_mask = passage_mask[:, passage_mask.sum(dim=0) > 0]

        # Shape: (batch_size, 4)
        action_logits = self._action_predictor(pooled_output)
        # Shape: (batch_size, passage_len, 2)       
        logits = self._span_predictor(passage_representation)
        span_start_logits, span_end_logits = logits.split(1, dim=-1)
        # Shape: (batch_size, passage_len)        
        span_start_logits = span_start_logits.squeeze(-1)
        # Shape: (batch_size, passage_len)        
        span_end_logits = span_end_logits.squeeze(-1)

        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        best_span = get_best_span(span_start_logits, span_end_logits)

        output_dict = {
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "span_end_logits": span_end_logits,
                "span_end_probs": span_end_probs,
                "best_span": best_span,
                "bert_output": bert_output,
                "pooled_output": pooled_output,
                "passage_representation": passage_representation
                }

        # Compute the loss for training.
        if span_start is not None:
            mask = (label.squeeze(-1) == self.vocab.get_token_index('More', namespace="labels")).float()
            span_loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start.squeeze(-1), reduction='none')
            if mask.sum() > 1e-7:
                self._span_start_accuracy(span_start_logits, span_start.squeeze(-1), mask)
            span_loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end.squeeze(-1), reduction='none')
            if mask.sum() > 1e-7:
                self._span_end_accuracy(span_end_logits, span_end.squeeze(-1), mask)
            span_acc_mask = mask.unsqueeze(1).expand(-1, 2).long()
            if span_acc_mask.sum() > 1e-7:
                self._span_accuracy(best_span, torch.cat([span_start, span_end], dim=1), span_acc_mask)
            span_loss = (span_loss * mask).sum() / (mask.sum() + 1e-6)

            action_loss = cross_entropy(action_logits, label.squeeze(-1))
            self._action_accuracy(action_logits, label.squeeze(-1))
            output_dict['loss'] = span_loss + 5 * action_loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            predictions = action_logits.cpu().data.numpy()
            argmax_indices = numpy.argmax(predictions, axis=-1)
            labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
            output_dict['label'] = labels
            batch_size = bert_output.size(0)
            for i in range(batch_size):
                bert_input_str = metadata[i]['original_bert_input']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = bert_input_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                # answer_texts = metadata[i].get('answer_texts', [])
                # if answer_texts and metadata[i]['action'] not in ['Yes', 'No', 'Irrelevant']:
                #     self._squad_metrics(best_span_string, answer_texts)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        try:
            exact_match, f1_score = self._squad_metrics.get_metric(reset)
        except ZeroDivisionError:
            exact_match, f1_score = 0, 0
        try:
            action_acc = self._action_accuracy.get_metric(reset)
        except ZeroDivisionError:
            action_acc = 0
        try:
            start_acc = self._span_start_accuracy.get_metric(reset)
        except ZeroDivisionError:
            start_acc = 0
        try:
            end_acc = self._span_end_accuracy.get_metric(reset)
        except ZeroDivisionError:
            end_acc = 0
        try:
            span_acc = self._span_accuracy.get_metric(reset)
        except ZeroDivisionError:
            span_acc = 0
        agg_metric = span_acc + action_acc * 0.45                                            
        return {
                'action_acc': action_acc,
                'start_acc': start_acc,
                'end_acc': end_acc,
                'span_acc': span_acc,
                'agg_metric': agg_metric
                }

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
        # We call the inputs "logits" - they could either be unnormalized logits or normalized log
        # probabilities.  A log_softmax operation is a constant shifting of the entire logit
        # vector, so taking an argmax over either one gives the same result.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                              device=device)).log().unsqueeze(0)
        valid_span_log_probs = span_log_probs + span_log_mask

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length
        return torch.stack([span_start_indices, span_end_indices], dim=-1)

    def wordpiece_to_tokens(self, tensor_, offsets):
        "Converts (bsz, orig_seq_len) to (bsz, seq_len) by indexing."        
        batch_size = tensor_.size(0)
        range_vector = util.get_range_vector(batch_size, device=util.get_device_of(tensor_)).unsqueeze(1)
        reduced_tensor = tensor_[range_vector, offsets]
        return reduced_tensor
