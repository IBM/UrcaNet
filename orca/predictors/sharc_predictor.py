from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('sharc_predictor')
class ShARCPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """

        rule_text = json_dict['snippet']
        question = json_dict['question']
        scenario = json_dict['scenario']
        history = json_dict['history']         

        return self._dataset_reader.text_to_instance(rule_text, question,\
                                                    scenario, history)