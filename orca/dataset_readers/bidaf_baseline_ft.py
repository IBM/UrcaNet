import json
import logging
from typing import Dict, List, Tuple, Optional

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from difflib import SequenceMatcher

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("bidaf_baseline_ft")
class BiDAFBaselineFTReader(DatasetReader):
    """
    Reads a JSON-formatted ShARC file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question`` a ``TextField``, which in our case is q || f1 ? a1 || ... || fm ? am,
    ``passage``, another ``TextField`` which is rule text in our case, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``. We also add a
    ``MetadataField`` that stores the utterance ID, tree ID, source URL, scenario and the gold answer,
    the original passage text, gold answer strings, and token offsets into the original passage,
    accessible as ``metadata['utterance_id']``, ``metadata['tree_id']``, ``metadata['source_url']``,
    ``metadata['scenario']``, ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.

    We also support limiting the maximum length for both passage and question. However, some gold
    answer spans may exceed the maximum passage length, which will cause error in making instances.
    We simply skip these spans to avoid errors. If all of the gold answer spans of an example
    are skipped, during training, we will skip this example. During validating or testing, since
    we cannot skip examples, we use the last token as the pseudo gold answer span instead. The
    computed loss will not be accurate as a result. But this will not affect the answer evaluation,
    because we keep all the original gold answer texts.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    skip_invalid_examples: ``bool``, optional (default=False)
        if this is true, we will skip those invalid examples
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 skip_invalid_examples: bool = True,
                 min_span_length: int = 3) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_invalid_examples = skip_invalid_examples
        self.min_span_length = min_span_length


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

    def find_answer_span(self, rule, answer):
        rule, answer = rule.lower(), answer.lower()
        rule = [token.text for token in self._tokenizer.tokenize(rule)]
        answer = [token.text for token in self._tokenizer.tokenize(answer)]

        sequenceMatcher = SequenceMatcher(None, rule, answer, autojunk=False)
        match = sequenceMatcher.find_longest_match(0, len(rule), 0, len(answer))
        if match.size < self.min_span_length:
            return None
        else:
            return match.a, match.a + match.size - 1

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

        passage_text = rule_text
        question_text = question + ' @@||@@ ' + scenario 
        for follow_up_qna in history:
            question_text += ' @@||@@ '
            question_text += follow_up_qna['follow_up_question']
            question_text += ' @@?@@ '
            question_text += follow_up_qna['follow_up_answer']
        
        passage_tokens = self._tokenizer.tokenize(passage_text)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_tokens = self._tokenizer.tokenize(question_text)
        # if self.passage_length_limit is not None:
        #     passage_tokens = passage_tokens[: self.passage_length_limit]
        # if self.question_length_limit is not None:
        #     question_tokens = question_tokens[: self.question_length_limit]
        
        fields: Dict[str, Field] = {}
        passage_field = TextField(passage_tokens, self._token_indexers)
        fields['passage'] = passage_field
        fields['question'] = TextField(question_tokens, self._token_indexers)
        metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens],}

        if answer:
            if answer in ['Yes', 'No', 'Irrelevant']:
                action = answer
                fields['span_start'] = IndexField(0, passage_field)
                fields['span_end'] = IndexField(0, passage_field)
            else:
                action = 'More'
                #TODO: change rule_text if there is passage length limit
                token_span = self.find_answer_span(rule_text, answer)
                if token_span is None:
                    if self.skip_invalid_examples:
                        return None
                    else:
                        passage_tokens_len = len(passage_tokens)
                        token_span = (max(passage_tokens_len - self.min_span_length, 0), passage_tokens_len - 1) 

                fields['span_start'] = IndexField(token_span[0], passage_field)
                fields['span_end'] = IndexField(token_span[1], passage_field)

                answer_text = passage_text[passage_offsets[token_span[0]][0]: passage_offsets[token_span[1]][1]]
                answer_texts = [answer_text]
                
                metadata['answer_texts'] = answer_texts
                metadata['original_answer'] = answer
            metadata['action'] = action
            fields['label'] = LabelField(action)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)