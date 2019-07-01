from typing import Dict, List, Callable
import logging

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.wordpiece_indexer import _get_token_type_ids, PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

@TokenIndexer.register("bert-pretrained_hist_aug")
class PretrainedBertHistoryAugmentedIndexer(PretrainedBertIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Currently any inputs longer than this
        will be truncated. If this behavior is undesirable to you, you should
        consider filtering them out in your dataset reader.
    """

    @overrides
    def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512) -> None:

        super().__init__(pretrained_model=pretrained_model,
                         use_starting_offsets=use_starting_offsets,
                         do_lowercase=do_lowercase,
                         never_lowercase=never_lowercase,
                         max_pieces=max_pieces)

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # The array of wordpiece_ids to return.
        # Start with a copy of the start_piece_ids
        wordpiece_ids: List[int] = self._start_piece_ids[:]

        PADDING = 0
        history_encoding = [PADDING] * len(self._start_piece_ids)
        turn_encoding = [PADDING] * len(self._start_piece_ids)

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = len(wordpiece_ids) if self.use_starting_offsets else len(wordpiece_ids) - 1

        for token in tokens:
            # Lowercase if necessary
            text = (token.text.lower()
                    if self._do_lowercase and token.text not in self._never_lowercase
                    else token.text)
            token_wordpiece_ids = [self.vocab[wordpiece]
                                   for wordpiece in self.wordpiece_tokenizer(text)]
            token_history_encoding = token.pos_ if isinstance(token.pos_, int) else PADDING
            token_turn_encoding = token.tag_ if isinstance(token.tag_, int) else PADDING
            history_encoding += [token_history_encoding] * len(token_wordpiece_ids)
            turn_encoding += [token_turn_encoding] * len(token_wordpiece_ids)
            # If we have enough room to add these ids *and also* the end_token ids.
            if len(wordpiece_ids) + len(token_wordpiece_ids) + len(self._end_piece_ids) <= self.max_pieces:
                # For initial offsets, the current value of ``offset`` is the start of
                # the current wordpiece, so add it to ``offsets`` and then increment it.
                if self.use_starting_offsets:
                    offsets.append(offset)
                    offset += len(token_wordpiece_ids)
                # For final offsets, the current value of ``offset`` is the end of
                # the previous wordpiece, so increment it and then add it to ``offsets``.
                else:
                    offset += len(token_wordpiece_ids)
                    offsets.append(offset)
                # And add the token_wordpiece_ids to the output list.
                wordpiece_ids.extend(token_wordpiece_ids)
            else:
                # TODO(joelgrus): figure out a better way to handle this
                logger.warning(f"Too many wordpieces, truncating: {[token.text for token in tokens]}")
                break

        # By construction, we still have enough room to add the end_token ids.
        wordpiece_ids.extend(self._end_piece_ids)
        history_encoding += [PADDING] * len(self._end_piece_ids)
        turn_encoding += [PADDING] * len(self._end_piece_ids)
        # Constructing `token_type_ids` by `self._separator`
        token_type_ids = _get_token_type_ids(wordpiece_ids,
                                             self._separator_ids)

        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        return {
                index_name: wordpiece_ids,
                f"{index_name}-offsets": offsets,
                f"{index_name}-type-ids": token_type_ids,
                "mask": mask,
                "history_encoding": history_encoding,
                "turn_encoding": turn_encoding
        }
