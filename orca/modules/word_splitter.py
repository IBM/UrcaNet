from typing import List
from overrides import overrides

from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import _remove_spaces, SpacyWordSplitter, WordSplitter

from spacy.tokenizer import Tokenizer
from spacy import util

def custom_tokenizer(nlp, never_split):
    cls = nlp.Defaults
    rules = cls.tokenizer_exceptions
    token_match = cls.token_match
    prefix_search = (
        util.compile_prefix_regex(cls.prefixes).search if cls.prefixes else None
    )
    suffix_search = (
        util.compile_suffix_regex(cls.suffixes).search if cls.suffixes else None
    )
    infix_finditer = (
        util.compile_infix_regex(cls.infixes).finditer if cls.infixes else None
    )
    vocab = nlp.vocab
    return Tokenizer(
        vocab,
        rules=rules,
        prefix_search=prefix_search,
        suffix_search=suffix_search,
        infix_finditer=infix_finditer,
        token_match=lambda x: token_match(x) or x in never_split,
    )

@WordSplitter.register('spacy_modified')
class SpacyWordSplitterModified(SpacyWordSplitter):
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False,
                 keep_spacy_tokens: bool = False,
                 never_split: List[str] = None) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)
        if never_split is not None:
            self.spacy.tokenizer = custom_tokenizer(self.spacy, never_split)
        self._keep_spacy_tokens = keep_spacy_tokens