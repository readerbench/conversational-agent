import numpy as np
import logging
import spacy
import string
from typing import Any, Text, Dict, List, Type

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    FEATURIZER_CLASS_ALIAS,
)
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE
from rasa.utils.tensorflow.constants import POOLING, MEAN_POOLING

from .syntactic_deps import deps, allowed_deps, precomputed_deps

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class SyntacticFeaturizer(DenseFeaturizer, GraphComponent):

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [SpacyTokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            **DenseFeaturizer.get_default_config(),
            # Specify what pooling operation should be used to calculate the vector of
            # the complete utterance. Available options: 'mean' and 'max'
            POOLING: MEAN_POOLING,
        }

    def __init__(self, config: Dict[Text, Any], name: Text) -> None:
        """Initializes SpacyFeaturizer."""
        super().__init__(name, config)
        self.pooling_operation = self._config[POOLING]

        # Initialize spaCy model used for syntactic-semantic parsing
        self.nlp_spacy = spacy.load('./models/spacy-syntactic')

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config, execution_context.node_name)

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes incoming messages and computes and sets features."""
        for message in messages:
            self._set_spacy_features(message, TEXT)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place.

        Args:
          training_data: Training data.

        Returns:
          Same training data after processing.
        """
        self.process(training_data.training_examples)
        return training_data

    def _set_spacy_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Adds the spacy word vectors to the messages features."""
        text = message.get(TEXT)
        if not text:
            return

        pre_deps = precomputed_deps[text] if text in precomputed_deps else None
        if not pre_deps:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", text)
            return

        doc = self.nlp_spacy(text.lower())
        word_syntactic_deps = []
        for i, spacy_token in enumerate(doc):
            no_spec_chars = spacy_token.text.translate(str.maketrans('', '', string.punctuation))
            word_syntactic_deps.append(spacy_token.dep_ if no_spec_chars and spacy_token.dep_ in deps else '-')

        """Feature vector for a single document / sentence / tokens."""
        a = np.array([deps.index(t) for t in word_syntactic_deps])
        sequence_features = np.zeros((a.size, len(deps)))
        sequence_features[np.arange(a.size), a] = 1
        sequence_features[:, 0] = 0

        sentence_features = self.aggregate_sequence_features(sequence_features, self.pooling_operation)
        # print(text, sentence_features)

        final_sentence_features = Features(
            sentence_features,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
