from difflib import SequenceMatcher
import json
import logging
import numpy as np
import regex
from spacy.lang.en.stop_words import STOP_WORDS

from typing import Dict, List, Tuple, Optional
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField, SequenceLabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedBertIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("bert_qa")
class BertQAReader(DatasetReader):
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
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy: bool = False,
                 add_scenario: bool = True,
                 add_history: bool = True,
                 max_context_length = 6) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.add_scenario = add_scenario
        self.add_history = add_history
        self.max_context_length = max_context_length
        self.lcs_cache = {}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        for utterance in dataset:
            rule = utterance['snippet']
            question = utterance['question']
            scenario = utterance['scenario']
            history = utterance['history']
            if 'answer' in utterance:
                answer = utterance['answer']
            if 'evidence' in utterance:
                evidence = utterance['evidence']
            instance = self.text_to_instance(rule, question, scenario, history, answer, evidence)
            if instance is not None:
                yield instance

    @staticmethod
    def all_stop_words(tokens_list, span):
        for i in range(span[0], span[1] + 1):
            if tokens_list[i].lower() not in STOP_WORDS:
                return False
        else:
            return True
    
    @staticmethod
    def find_closest_element_ix(list_, number):
        "Returns the index of closest element in sorted list"
        for i, element in enumerate(list_):
            if element == number:
                return i
            elif element > number:
                break
        just_larger_ix = i
        if just_larger_ix == 0 or list_[just_larger_ix] - number < number - list_[just_larger_ix - 1]:
            return just_larger_ix
        else:
            return just_larger_ix - 1

    def find_lcs(self, text1, text2, tokenizer_fn, min_length=3, fuzzy_matching=True, filter_stop_words=True):
        """
        Returns start and end (token) index of longest common subsequence (of text1 and text2) in text1.
        If fuzzy matching is True, it is used when lcs is less than min_length.
        If filter_stop_words is True, in case the found span contains only stop words, None is returned. 
        """ 

        args = (text1, text2, tokenizer_fn, min_length, fuzzy_matching)
        if args in self.lcs_cache:
            return self.lcs_cache[args] 

        text1_tokens = [token.text.lower() for token in tokenizer_fn(text1)]
        text1_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenizer_fn(text1)]
        text2_tokens = [token.text.lower() for token in tokenizer_fn(text2)]

        sequence_matcher = SequenceMatcher(None, text1_tokens, text2_tokens, autojunk=False)
        lcs_match = sequence_matcher.find_longest_match(0, len(text1_tokens), 0, len(text2_tokens))
        lcs_span = lcs_match.a, lcs_match.a + lcs_match.size - 1
        regex_span = None

        if (lcs_match.size < min_length or self.all_stop_words(text1_tokens, lcs_span)) and fuzzy_matching:
                pattern = r'(?:\b' + regex.escape(text2.lower()) + r'\b){i<=6,d<=20}'
                regex_match = regex.search(pattern, text1.lower(), regex.BESTMATCH)
                if regex_match:
                    start_token_ix = self.find_closest_element_ix([offset[0] for offset in text1_offsets], regex_match.span()[0])
                    end_token_ix = self.find_closest_element_ix([offset[1] for offset in text1_offsets], regex_match.span()[1])
                    regex_span = start_token_ix, end_token_ix
                    
        if regex_span is not None and not self.all_stop_words(text1_tokens, regex_span):
            self.lcs_cache[args] = regex_span
        elif lcs_match.size > 0 and not self.all_stop_words(text1_tokens, lcs_span):
            self.lcs_cache[args] = lcs_span
        else:
            self.lcs_cache[args] = None
        
        return self.lcs_cache[args]

    def tokenize_and_add_encodings(self, bert_input, passage_text, history=None, evidence=None,
                                   add_history_encoding=False, add_turn_encoding=False, add_scenario_encoding=False):

        if (add_history_encoding or add_turn_encoding) and history is None:
            raise ValueError("History must be passed to add history encoding.") 
        if add_scenario_encoding and evidence is None:
            raise ValueError("Evidence must be passed to add scenario encoding.") 
        
        NOT_ANSWERED, ANSWERED_YES, ANSWERED_NO = 1, 2, 3 # for history and scenario encoding 
        TURN_ENCODING_NOT_ASKED = self.max_context_length + 1

        passage_tokens_len = len(self._tokenizer.tokenize(passage_text))

        history_encoding = np.ones(passage_tokens_len, dtype=np.int64) * NOT_ANSWERED
        turn_encoding = np.ones(passage_tokens_len, dtype=np.int64) * TURN_ENCODING_NOT_ASKED
        scenario_encoding = np.ones(passage_tokens_len, dtype=np.int64) * NOT_ANSWERED 

        if add_history_encoding or add_turn_encoding:
            turn_recency = 1
            for followup_qa in reversed(history):
                if 'follow_up_question' not in followup_qa or 'follow_up_answer' not in followup_qa: # dataset issue
                    continue 
                followup_ques = followup_qa['follow_up_question']
                followup_ans = followup_qa['follow_up_answer']
                token_span = self.find_lcs(passage_text, followup_ques, tokenizer_fn=self._tokenizer.tokenize)
                if token_span is not None:
                    start_ix = token_span[0]
                    end_ix = token_span[1] + 1 # exclusive
                    if followup_ans == 'Yes':
                        history_encoding[start_ix: end_ix] = ANSWERED_YES
                    else: 
                        history_encoding[start_ix: end_ix] = ANSWERED_NO
                    turn_encoding[start_ix: end_ix] = min(turn_recency, self.max_context_length)
                turn_recency += 1 
        
        if add_scenario_encoding:
            for followup_qa in evidence:
                if 'follow_up_question' not in followup_qa or 'follow_up_answer' not in followup_qa: # dataset issue
                    continue 
                followup_ques = followup_qa['follow_up_question']
                followup_ans = followup_qa['follow_up_answer']
                token_span = self.find_lcs(passage_text, followup_ques, tokenizer_fn=self._tokenizer.tokenize)
                if token_span is not None:
                    start_ix = token_span[0]
                    end_ix = token_span[1] + 1 # exclusive
                    if followup_ans == 'Yes':
                        scenario_encoding[start_ix: end_ix] = ANSWERED_YES
                    else:
                        scenario_encoding[start_ix: end_ix] = ANSWERED_NO
    
        bert_input_tokens = self._tokenizer.tokenize(bert_input)

        if add_history_encoding:
            for i, token in enumerate(bert_input_tokens[:passage_tokens_len]):
                bert_input_tokens[i] = token._replace(pos_=int(history_encoding[i]))

        if add_turn_encoding:
            for i, token in enumerate(bert_input_tokens[:passage_tokens_len]):
                bert_input_tokens[i] = token._replace(tag_=int(turn_encoding[i]))

        if add_scenario_encoding:
            for i, token in enumerate(bert_input_tokens[:passage_tokens_len]):
                bert_input_tokens[i] = token._replace(dep_=int(scenario_encoding[i]))

        return bert_input_tokens

    @overrides
    def text_to_instance(self,  # type: ignore
                        rule: str,
                        question: str,
                        scenario: str,
                        history: List[Dict[str, str]],
                        answer: str = None,
                        evidence: List[Dict[str, str]] = None) -> Optional[Instance]:
        
        passage_text = rule + ' [SEP]'
        question_text = question + ' @qe@'
        if self.add_scenario:
            question_text += ' @ss@ ' + scenario + ' @se'
        if self.add_history:
            question_text += ' @hs@'
            for follow_up_qna in history[:self.max_context_length]:
                question_text += ' @qs@'
                question_text += ' ' + follow_up_qna['follow_up_question']
                question_text += ' ' + follow_up_qna['follow_up_answer']
            question_text += ' @he@'
        
        bert_input = passage_text + ' ' + question_text
        bert_input_tokens = self.tokenize_and_add_encodings(bert_input, passage_text, history=history,
                                                            add_history_encoding=True, add_turn_encoding=True)

        passage_text_sim = rule + ' [SEP]'
        question_text_sim = '@ss@ ' + scenario + ' @se@'
        sim_bert_input = passage_text_sim + ' ' + question_text_sim
        if evidence is not None:
            sim_bert_input_tokens = self.tokenize_and_add_encodings(sim_bert_input, passage_text_sim,
                                                                    evidence=evidence, add_scenario_encoding=True)
        else:
            sim_bert_input_tokens = self._tokenizer.tokenize(sim_bert_input)

        assert passage_text == passage_text_sim

        passage_tokens = self._tokenizer.tokenize(passage_text)
        passage_field = TextField(passage_tokens, self._token_indexers)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        
        fields: Dict[str, Field] = {}
        fields['bert_input'] = TextField(bert_input_tokens, self._token_indexers)
        fields['sim_bert_input'] = TextField(sim_bert_input_tokens, self._token_indexers)

        metadata = {'rule': rule, 'question': question, 'history': history, 'scenario': scenario,
                    'passage_text': passage_text, 'token_offsets': passage_offsets, 'passage_tokens': passage_tokens}
        if evidence:
            metadata['evidence'] = evidence

        if answer: # true while training, validating
            if answer in ['Yes', 'No', 'Irrelevant']:
                fields['span_start'] = IndexField(0, passage_field) # This doesn't matter as we mask the loss.
                fields['span_end'] = IndexField(0, passage_field)
            else:
                token_span = self.find_lcs(passage_text, answer, tokenizer_fn=self._tokenizer.tokenize)
                if token_span is None:
                    return None
                fields['span_start'] = IndexField(token_span[0], passage_field)
                fields['span_end'] = IndexField(token_span[1], passage_field)
                gold_span = passage_text[passage_offsets[token_span[0]][0]: passage_offsets[token_span[1]][1]]
                metadata['gold_span'] = gold_span
            metadata['answer'] = answer
            action = answer if answer in ['Yes', 'No', 'Irrelevant'] else 'More'
            metadata['action'] = action
            fields['label'] = LabelField(action)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)