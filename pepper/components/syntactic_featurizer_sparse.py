from __future__ import annotations
from collections import OrderedDict
import logging
import scipy.sparse
import spacy
import string
import numpy as np
from typing import Any, Dict, Text, List, Tuple, Callable, Set, Optional, Type, Union

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.featurizers.sparse_featurizer.sparse_featurizer import SparseFeaturizer
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT
from rasa.shared.exceptions import InvalidConfigException
import rasa.shared.utils.io
import rasa.utils.io

from .syntactic_deps import deps, allowed_deps, precomputed_deps

logger = logging.getLogger(__name__)

END_OF_SENTENCE = "EOS"
BEGIN_OF_SENTENCE = "BOS"

FEATURES = "features"


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class SyntacticFeaturizer(SparseFeaturizer, GraphComponent):
    """Extracts and encodes lexical syntactic features.

    Given a sequence of tokens, this featurizer produces a sequence of features
    where the `t`-th feature encodes lexical and syntactic information about the `t`-th
    token and it's surrounding tokens.

    In detail: The lexical syntactic features can be specified via a list of
    configurations `[c_0, c_1, ..., c_n]` where each `c_i` is a list of names of
    lexical and syntactic features (e.g. `low`, `suffix2`, `digit`).
    For a given tokenized text, the featurizer will consider a window of size `n`
    around each token and evaluate the given list of configurations as follows:
    - It will extract the features listed in `c_m` where `m = (n-1)/2` if n is even and
      `n/2` from token `t`
    - It will extract the features listed in `c_{m-1}`,`c_{m-2}` ... ,  from the last,
      second to last, ... token before token `t`, respectively.
    - It will extract the features listed `c_{m+1}`, `c_{m+1}`, ... for the first,
      second, ... token `t`, respectively.
    It will then combine all these features into one feature for position `t`.

    Example:
      If we specify `[['low'], ['upper'], ['prefix2']]`, then for each position `t`
      the `t`-th feature will encode whether the token at position `t` is upper case,
      where the token at position `t-1` is lower case and the first two characters
      of the token at position `t+1`.
    """

    FILENAME_FEATURE_TO_IDX_DICT = "synt_feature_to_idx_dict.pkl"

    # NOTE: "suffix5" of the token "is" will be "is". Hence, when combining multiple
    # prefixes, short words will be represented/encoded repeatedly.
    _FUNCTION_DICT: Dict[Text, Callable[[Token], Union[Text, bool, None]]] = {
        "syntactic_dep": lambda token: token.text,
    }

    SUPPORTED_FEATURES = sorted(
        set(_FUNCTION_DICT.keys()).union([END_OF_SENTENCE, BEGIN_OF_SENTENCE])
    )

    @classmethod
    def _extract_raw_features_from_token(
            cls, feature_name: Text, token: Token, token_position: int, num_tokens: int
    ) -> Text:
        """Extracts a raw feature from the token at the given position.

        Args:
          feature_name: the name of a supported feature
          token: the token from which we want to extract the feature
          token_position: the position of the token inside the tokenized text
          num_tokens: the total number of tokens in the tokenized text
        Returns:
          the raw feature value as text
        """
        if feature_name not in cls.SUPPORTED_FEATURES:
            raise InvalidConfigException(
                f"Configured feature '{feature_name}' not valid. Please check "
                f"'{DOCS_URL_COMPONENTS}' for valid configuration parameters."
            )
        if feature_name == END_OF_SENTENCE:
            return str(token_position == num_tokens - 1)
        if feature_name == BEGIN_OF_SENTENCE:
            return str(token_position == 0)
        return str(cls._FUNCTION_DICT[feature_name](token))

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            **SparseFeaturizer.get_default_config(),
            FEATURES: [
                ["syntactic_dep"]
            ],
        }

    def __init__(
            self,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            feature_to_idx_dict: Optional[Dict[Tuple[int, Text], Dict[Text, int]]] = None,
    ) -> None:
        """Instantiates a new `LexicalSyntacticFeaturizer` instance."""
        super().__init__(execution_context.node_name, config)
        # graph component
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context
        # featurizer specific
        self._feature_config = self._config[FEATURES]
        self._set_feature_to_idx_dict(
            feature_to_idx_dict or {}, check_consistency_with_config=True
        )

        # Initialize spaCy model used for syntactic-semantic parsing
        self.nlp_spacy = spacy.load('./models/spacy-syntactic')

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        if FEATURES not in config:
            return  # will be replaced with default
        feature_config = config[FEATURES]
        message = (
            f"Expected configuration of `features` to be a list of lists that "
            f"that contain names of lexical and syntactic features "
            f"(i.e. {cls.SUPPORTED_FEATURES}). "
            f"Received {feature_config} instead. "
        )
        try:
            configured_feature_names = set(
                feature_name
                for pos_config in feature_config
                for feature_name in pos_config
            )
        except TypeError as e:
            raise InvalidConfigException(message) from e
        if configured_feature_names.difference(cls.SUPPORTED_FEATURES):
            raise InvalidConfigException(message)

    def _set_feature_to_idx_dict(
            self,
            feature_to_idx_dict: Dict[Tuple[int, Text], Dict[Text, int]],
            check_consistency_with_config: bool = False,
    ) -> None:
        """Sets the "feature" to index mapping.

        Here, "feature" denotes the combination of window position, feature name,
        and feature_value.

        Args:
          feature_to_idx_dict: mapping from tuples of window position and feature name
            to a mapping from feature values to indices
          check_consistency_with_config: whether the consistency with the current
            `self.config` should be checked
        """
        self._feature_to_idx_dict = feature_to_idx_dict
        self._number_of_features = sum(
            [
                len(feature_values.values())
                for feature_values in self._feature_to_idx_dict.values()
            ]
        )
        if check_consistency_with_config:
            known_features = set(self._feature_to_idx_dict.keys())
            not_in_config = known_features.difference(
                (
                    (window_idx, feature_name)
                    for window_idx, feature_names in enumerate(self._feature_config)
                    for feature_name in feature_names
                )
            )
            if not_in_config:
                rasa.shared.utils.io.raise_warning(
                    f"A feature to index mapping has been loaded that does not match "
                    f"the configured features. The given mapping configures "
                    f" (position in window, feature_name): {not_in_config}. "
                    f" These are not specified in the given config "
                    f" {self._feature_config}. "
                    f"Continuing with constant values for these features. "
                )

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the featurizer.

        Args:
          training_data: the training data

        Returns:
           the resource from which this trained component can be loaded
        """
        feature_to_idx_dict = self._create_feature_to_idx_dict(training_data)
        self._set_feature_to_idx_dict(feature_to_idx_dict=feature_to_idx_dict)
        if not self._feature_to_idx_dict:
            rasa.shared.utils.io.raise_warning(
                "No lexical syntactic features could be extracted from the training "
                "data. In order for this component to work you need to define "
                "`features` that can be found in the given training data."
            )
        self.persist()
        return self._resource

    def _deps_for_text(self, text: str):
        doc = self.nlp_spacy(text.lower())
        pre_deps = precomputed_deps[text] if text in precomputed_deps else None
        if not pre_deps:
            raise Exception(f"No manually annotated syntactic features were defined for sentence <{text}>")

        word_syntactic_deps = []
        for i, spacy_token in enumerate(doc):
            no_spec_chars = spacy_token.text.translate(str.maketrans('', '', string.punctuation))
            # word_syntactic_deps.append(spacy_token.dep_ if no_spec_chars and spacy_token.dep_ in deps else '-')
            word_syntactic_deps.append(pre_deps[i] if pre_deps else '-')

        return word_syntactic_deps

    def _create_feature_to_idx_dict(
            self, training_data: TrainingData
    ) -> Dict[Tuple[int, Text], Dict[Text, int]]:
        """Create a nested dictionary of all feature values.

        Returns:
           a nested mapping that maps from tuples of positions (in the window) and
           supported feature names to "raw feature to index" mappings, i.e.
           mappings that map the respective raw feature values to unique indices
           (where `unique` means unique with respect to all indices in the
           *nested* mapping)
        """
        # collect all raw feature values
        feature_vocabulary: Dict[Tuple[int, Text], Set[Text]] = dict()
        for example in training_data.training_examples:
            tokens = example.get(TOKENS_NAMES[TEXT], [])

            text = example.get(TEXT)
            if text:
                sentence_features = self._map_tokens_to_raw_features(tokens, text)

                for token_features in sentence_features:
                    for position_and_feature_name, feature_value in token_features.items():
                        feature_vocabulary.setdefault(position_and_feature_name, set()).add(feature_value)

        # assign a unique index to each feature value
        return self._build_feature_to_index_map(feature_vocabulary)

    def _map_tokens_to_raw_features(self, tokens: List[Token], text: str) -> List[Dict[Tuple[int, Text], Text]]:
        """Extracts the raw feature values.

        Args:
          tokens: a tokenized text
        Returns:
          a list of feature dictionaries for each token in the given list
          where each feature dictionary maps a tuple containing
          - a position (in the window) and
          - a supported feature name
          to the corresponding raw feature value
        """
        sentence_features = []

        syntactic_deps = self._deps_for_text(text)
        # print(list(map(lambda t: t.text, tokens)), f"\"{text}\"", syntactic_deps)

        # in case of an even number we will look at one more word before,
        # e.g. window size 4 will result in a window range of
        # [-2, -1, 0, 1] (0 = current word in sentence)
        window_size = len(self._feature_config)
        half_window_size = window_size // 2
        window_range = range(-half_window_size, half_window_size + window_size % 2)
        assert len(window_range) == window_size

        for anchor in range(len(tokens)):

            token_features: Dict[Tuple[int, Text], Text] = {}

            for window_position, relative_position in enumerate(window_range):
                absolute_position = anchor + relative_position

                # skip, if current_idx is pointing to a non-existing token
                if absolute_position < 0 or absolute_position >= len(tokens):
                    continue

                token = tokens[absolute_position]
                for feature_name in self._feature_config[window_position]:
                    token_features[(window_position, feature_name)] = syntactic_deps[absolute_position]

            sentence_features.append(token_features)

        return sentence_features

    @staticmethod
    def _build_feature_to_index_map(
            feature_vocabulary: Dict[Tuple[int, Text], Set[Text]]
    ) -> Dict[Tuple[int, Text], Dict[Text, int]]:
        """Creates a nested dictionary for mapping raw features to indices.

        Args:
          feature_vocabulary: a mapping from tuples of positions (in the window) and
            supported feature names to the set of possible feature values
        Returns:
           a nested mapping that maps from tuples of positions (in the window) and
           supported feature names to "raw feature to index" mappings, i.e.
           mappings that map the respective raw feature values to unique indices
           (where `unique` means unique with respect to all indices in the
           *nested* mapping)
        """
        # Note that this will only sort the top level keys - and we keep
        # doing it to ensure consistently with what was done before)
        ordered_feature_vocabulary: OrderedDict[Tuple[int, Text], Set[Text]] = \
            OrderedDict(sorted(feature_vocabulary.items()))

        # create the nested mapping
        feature_to_idx_dict: Dict[Tuple[int, Text], Dict[Text, int]] = {}
        offset = 0
        for (
                position_and_feature_name,
                feature_values,
        ) in ordered_feature_vocabulary.items():
            sorted_feature_values = sorted(feature_values)
            feature_to_idx_dict[position_and_feature_name] = {
                feature_value: feature_idx
                for feature_idx, feature_value in enumerate(
                    sorted_feature_values, start=offset
                )
            }
            offset += len(feature_values)

        return feature_to_idx_dict

    def process(self, messages: List[Message]) -> List[Message]:
        """Featurizes all given messages in-place.

        Args:
          messages: messages to be featurized.

        Returns:
          The same list with the same messages after featurization.
        """
        for message in messages:
            self._process_message(message)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place.

        Args:
          training_data: the training data

        Returns:
          same training data after processing
        """
        self.process(training_data.training_examples)
        return training_data

    def _process_message(self, message: Message) -> None:
        """Featurizes the given message in-place.

        Args:
          message: a message to be featurized
        """
        if not self._feature_to_idx_dict:
            rasa.shared.utils.io.raise_warning(
                f"The {self.__class__.__name__} {self._identifier} has not been "
                f"trained properly yet. "
                f"Continuing without adding features from this featurizer."
            )
            return
        tokens = message.get(TOKENS_NAMES[TEXT])
        text = message.get(TEXT)
        if text and tokens:
            sentence_features = self._map_tokens_to_raw_features(tokens, text)
            sparse_matrix = self._map_raw_features_to_indices(sentence_features)

            self.add_features_to_message(
                # FIXME: create sentence feature and make `sentence` non optional
                sequence=sparse_matrix,
                sentence=None,
                attribute=TEXT,
                message=message,
            )

    def _map_raw_features_to_indices(
            self, sentence_features: List[Dict[Tuple[int, Text], Any]]
    ) -> scipy.sparse.coo_matrix:
        """Converts the raw features to one-hot encodings.

        Requires the "feature" to index dictionary, i.e. the featurizer must have
        been trained.

        Args:
          sentence_features: a list of feature dictionaries where the `t`-th feature
            dictionary maps a tuple containing
            - a position (in the window) and
            - a supported feature name
            to the raw feature value extracted from the window around the `t`-th token.

        Returns:
           a sparse matrix where the `i`-th row is a multi-hot vector that encodes the
           raw features extracted from the window around the `i`-th token
        """
        rows = []
        cols = []
        shape = (len(sentence_features), self._number_of_features)
        for token_idx, token_features in enumerate(sentence_features):
            for position_and_feature_name, feature_value in token_features.items():
                mapping = self._feature_to_idx_dict.get(position_and_feature_name)
                if not mapping:
                    continue
                feature_idx = mapping.get(feature_value, -1)
                if feature_idx > -1:
                    rows.append(token_idx)
                    cols.append(feature_idx)
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.ones(len(rows))
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=shape)

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> SyntacticFeaturizer:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    @classmethod
    def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            **kwargs: Any,
    ) -> SyntacticFeaturizer:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_path:
                feature_to_idx_dict = rasa.utils.io.json_unpickle(
                    model_path / cls.FILENAME_FEATURE_TO_IDX_DICT,
                    encode_non_string_keys=True,
                )
                return cls(
                    config=config,
                    model_storage=model_storage,
                    resource=resource,
                    execution_context=execution_context,
                    feature_to_idx_dict=feature_to_idx_dict,
                )
        except ValueError:
            logger.debug(
                f"Failed to load `{cls.__class__.__name__}` from model storage. "
                f"Resource '{resource.name}' doesn't exist."
            )
            return cls(
                config=config,
                model_storage=model_storage,
                resource=resource,
                execution_context=execution_context,
            )

    def persist(self) -> None:
        """Persist this model (see parent class for full docstring)."""
        if not self._feature_to_idx_dict:
            return None

        with self._model_storage.write_to(self._resource) as model_path:
            rasa.utils.io.json_pickle(
                model_path / self.FILENAME_FEATURE_TO_IDX_DICT,
                self._feature_to_idx_dict,
                encode_non_string_keys=True,
            )
