import logging
import yaml
from pathlib import Path
from typing import Any, List, Type, Text, Dict, Optional

from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.classifiers.diet_classifier import DIETClassifier, EntityTagSpec
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.models import RasaModel

from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
)

GENERIC_INTENT = 'microworld'
GENERIC_INTENT_RANKING = f'{GENERIC_INTENT}_ranking'

logger = logging.getLogger(__name__)


def get_microworlds():
    """ Load the microworlds registered in the config file. """

    with open('config.yml') as config_file:
        # The FullLoader parameter handles the conversion from YAML scalar values to Python dict
        config = yaml.load(config_file, Loader=yaml.FullLoader)

        microworld_clf_config = next(c for c in config['pipeline'] if c['name'].endswith("MicroworldClassifier"))
        microworlds = list(microworld_clf_config['microworlds'].keys())

    logger.info("Registered microworlds: " + str(microworlds))
    return microworlds


class MicroworldClassifier(IntentClassifier):
    """
    DIET-based classifier that detects the microworld-specific intent and entities.
    """

    defaults = DIETClassifier.defaults
    microworlds = get_microworlds()

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None,
                 index_label_id_mapping: Optional[Dict[int, Text]] = None,
                 entity_tag_specs: Optional[List[EntityTagSpec]] = None, model: Optional[RasaModel] = None) -> None:
        super().__init__(component_config)

        self.classifiers = {}

        for microworld in self.microworlds:
            if component_config and 'microworlds' in component_config:
                mw_component_config = component_config['microworlds'][microworld].copy()
                mw_component_config['name'] = 'DIETClassifier'
            else:
                mw_component_config = component_config

            self.classifiers[microworld] = DIETClassifier(mw_component_config,
                                                          index_label_id_mapping,
                                                          entity_tag_specs,
                                                          model)

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [IntentClassifier, Featurizer]

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return DIETClassifier.required_packages()

    def process(self, message: Message, **kwargs: Any) -> None:
        """
        Proxy component that dispatches the classification task to the classifier specific
        to the microworld the current message belongs to.

        :param message: The :class:`rasa.shared.nlu.training_data.message.Message` to process.
        :return None, just updates the message structure
        """

        # Select the classifier corresponding to the detected microworld
        generic_intent = message.get(INTENT).get(INTENT_NAME_KEY)

        # Rename the intent information extracted by the domain classifier
        message.set(GENERIC_INTENT, message.get(INTENT), add_to_output=True)
        message.set(GENERIC_INTENT_RANKING, message.get(INTENT_RANKING_KEY), add_to_output=True)

        # Call the classifier, passing the arguments unchanged
        self.classifiers[generic_intent.split('.')[0]].process(message, **kwargs)

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        for microworld in self.microworlds:
            # Extract the training examples corresponding to the current microworld
            mw_training_data = training_data.filter_training_examples(
                lambda msg: 'intent' not in msg.data or
                            msg.data['intent'].startswith(microworld)
            )

            logger.info(f"Training DIETClassifier for microworld: {microworld}")
            self.classifiers[microworld].train(mw_training_data, config, **kwargs)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        for microworld in self.microworlds:
            self.classifiers[microworld].persist(f"{file_name}_{microworld}", model_dir)

        return {"file": file_name}

    @classmethod
    def load(
            cls,
            meta: Dict[Text, Any],
            model_dir: Text = None,
            model_metadata: Metadata = None,
            cached_component: Optional["MicroworldClassifier"] = None,
            **kwargs: Any,
    ) -> "MicroworldClassifier":
        """Loads the trained model from the provided directory."""

        mw_classif = cls()
        component_name = meta['file']
        for microworld in cls.microworlds:
            meta['file'] = f"{component_name}_{microworld}"
            mw_classif.classifiers[microworld] = DIETClassifier.load(meta, model_dir, model_metadata, None, **kwargs)

        return mw_classif
