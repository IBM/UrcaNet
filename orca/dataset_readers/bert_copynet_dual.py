import logging
from typing import List, Dict, Optional
import json

import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from difflib import SequenceMatcher

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("bert_copynet_dual")
class BertCopyNetDualDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``CopyNet`` model, or any model with a matching API.

    The expected format for each input line is: <source_sequence_string><tab><target_sequence_string>.
    An instance produced by ``CopyNetDatasetReader`` will containing at least the following fields:

    - ``source_tokens``: a ``TextField`` containing the tokenized source sentence,
       including the ``START_SYMBOL`` and ``END_SYMBOL``.
       This will result in a tensor of shape ``(batch_size, source_length)``.

    - ``source_token_ids``: an ``ArrayField`` of size ``(batch_size, trimmed_source_length)``
      that contains an ID for each token in the source sentence. Tokens that
      match at the lowercase level will share the same ID. If ``target_tokens``
      is passed as well, these IDs will also correspond to the ``target_token_ids``
      field, i.e. any tokens that match at the lowercase level in both
      the source and target sentences will share the same ID. Note that these IDs
      have no correlation with the token indices from the corresponding
      vocabulary namespaces.

    - ``source_to_target``: a ``NamespaceSwappingField`` that keeps track of the index
      of the target token that matches each token in the source sentence.
      When there is no matching target token, the OOV index is used.
      This will result in a tensor of shape ``(batch_size, trimmed_source_length)``.

    - ``metadata``: a ``MetadataField`` which contains the source tokens and
      potentially target tokens as lists of strings.

    When ``target_string`` is passed, the instance will also contain these fields:

    - ``target_tokens``: a ``TextField`` containing the tokenized target sentence,
      including the ``START_SYMBOL`` and ``END_SYMBOL``. This will result in
      a tensor of shape ``(batch_size, target_length)``.

    - ``target_token_ids``: an ``ArrayField`` of size ``(batch_size, target_length)``.
      This is calculated in the same way as ``source_token_ids``.

    See the "Notes" section below for a description of how these fields are used.

    Parameters
    ----------
    target_namespace : ``str``, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the NamespaceSwappingField.
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.

    Notes
    -----
    By ``source_length`` we are referring to the number of tokens in the source
    sentence including the ``START_SYMBOL`` and ``END_SYMBOL``, while
    ``trimmed_source_length`` refers to the number of tokens in the source sentence
    *excluding* the ``START_SYMBOL`` and ``END_SYMBOL``, i.e.
    ``trimmed_source_length = source_length - 2``.

    On the other hand, ``target_length`` is the number of tokens in the target sentence
    *including* the ``START_SYMBOL`` and ``END_SYMBOL``.

    In the context where there is a ``batch_size`` dimension, the above refer
    to the maximum of their individual values across the batch.

    In regards to the fields in an ``Instance`` produced by this dataset reader,
    ``source_token_ids`` and ``target_token_ids`` are primarily used during training
    to determine whether a target token is copied from a source token (or multiple matching
    source tokens), while ``source_to_target`` is primarily used during prediction
    to combine the copy scores of source tokens with the generation scores for matching
    tokens in the target namespace.
    """

    def __init__(self,
                 target_namespace: str,
                 bert_tokenizer: Tokenizer,
                 bert_token_indexers: Dict[str, TokenIndexer],
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._bert_tokenizer = bert_tokenizer
        self._bert_token_indexers = bert_token_indexers 
        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        if "tokens" not in self._source_token_indexers or \
                not isinstance(self._source_token_indexers["tokens"], SingleIdTokenIndexer):
            raise ConfigurationError("CopyNetDatasetReader expects 'source_token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        self._target_token_indexers: Dict[str, TokenIndexer] = {
                "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
        }

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        for utterance in dataset:
            utterance_id = utterance['utterance_id']
            tree_id = utterance['tree_id']
            source_url = utterance['source_url']
            rule_text = utterance['snippet']
            question = utterance['question']
            scenario = utterance['scenario']
            history = utterance['history']

            if 'answer' in utterance.keys():
                answer = utterance['answer']
            if 'evidence' in utterance.keys():
                evidence = utterance['evidence']
            
            instance = self.text_to_instance(rule_text, question, scenario, history,\
                                            utterance_id, tree_id, source_url,\
                                            answer, evidence)

            if instance is not None:
                yield instance

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    def find_answer_span(self, rule, answer, min_span_length):
        rule, answer = rule.lower(), answer.lower()
        rule = [token.text for token in self._bert_tokenizer.tokenize(rule)]
        answer = [token.text for token in self._bert_tokenizer.tokenize(answer)]

        sequenceMatcher = SequenceMatcher(None, rule, answer, autojunk=False)
        match = sequenceMatcher.find_longest_match(0, len(rule), 0, len(answer))
        if match.size < min_span_length:
            return None
        else:
            return match.a, match.a + match.size - 1

    def get_tokens_with_history_encoding(self, bert_input, history):
        NOT_ASKED = 1
        ASKED_ANS_YES = 2
        ASKED_ANS_NO = 3

        passage_text = bert_input.split('[SEP]')[0][:-1]
        passage_tokens_len = len(self._bert_tokenizer.tokenize(passage_text))
        history_encoding = [NOT_ASKED] * passage_tokens_len  
        for followup_qa in history:
            followup_ques = followup_qa['follow_up_question']
            followup_ans = followup_qa['follow_up_answer']
            token_span = self.find_answer_span(passage_text, followup_ques, 0)
            if token_span is not None:
                start_ix = token_span[0]
                end_ix = token_span[1] + 1 # exclusive
                span_size = end_ix - start_ix
                if followup_ans == 'Yes':
                    history_encoding[start_ix: end_ix] = [ASKED_ANS_YES] * span_size
                else:
                    history_encoding[start_ix: end_ix] = [ASKED_ANS_NO] * span_size
        
        bert_input_tokens = self._bert_tokenizer.tokenize(bert_input)
        for i, token in enumerate(bert_input_tokens[:passage_tokens_len]):
            bert_input_tokens[i] = token._replace(pos_=history_encoding[i])
        return bert_input_tokens

    @overrides
    def text_to_instance(self,  # type: ignore
                        rule_text: str,
                        question: str,
                        scenario: str,
                        history: List[Dict[str, str]],
                        utterance_id: str = None,
                        tree_id: str = None,
                        source_url: str = None,
                        answer: str = None,
                        evidence: List[Dict[str, str]] = None) -> Optional[Instance]:
        """
        Turn raw source string and target string into an ``Instance``.

        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional (default = None)

        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """

        # For CopyNet Model
        source_string = rule_text + ' [SEP]'
        target_string = answer
    
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        # tokenized_source.append(Token(END_SYMBOL)) ' @@SEP@@' acts as end symbol
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        # For Bert model
        passage_text1 = rule_text + ' [SEP]'
        question_text1 = question

        passage_text2 = rule_text + ' [SEP]'
        question_text2 = scenario

        bert_input1 = passage_text1 + ' ' + question_text1
        bert_input2 = passage_text2 + ' ' + question_text2

        bert_input_tokens1 = self.get_tokens_with_history_encoding(bert_input1, history)
        bert_input_tokens2 = self._bert_tokenizer.tokenize(bert_input2)
        bert_input_tokens1.insert(0, Token(START_SYMBOL))
        bert_input_tokens2.insert(0, Token(START_SYMBOL))
        fields_dict['bert_input1'] = TextField(bert_input_tokens1, self._bert_token_indexers)
        fields_dict['bert_input2'] = TextField(bert_input_tokens2, self._bert_token_indexers)
        meta_fields['passage_tokens1'] = self._bert_tokenizer.tokenize(passage_text1)
        meta_fields['passage_tokens2'] = self._bert_tokenizer.tokenize(passage_text2)

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
            
            action = 'More' if answer not in ['Yes', 'No', 'Irrelevant'] else answer
            fields_dict['label'] = LabelField(action)
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        meta_fields['rule_text'] = rule_text
        meta_fields['question'] = question
        meta_fields['scenario'] = scenario
        meta_fields['history'] = history
        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
