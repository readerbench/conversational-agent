from typing import Any, List, Type, Text, Dict
import logging
import yaml
from pathlib import Path

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

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


class MultiModelStorage(LocalModelStorage):
    def __init__(self, storage_path: Path, subfolder: str) -> None:
        """Creates storage (see parent class for full docstring)."""
        super().__init__(storage_path)
        self._subfolder = subfolder

    def _directory_for_resource(self, resource: Resource) -> Path:
        directory = self._storage_path / resource.name / self._subfolder

        if not directory.exists():
            directory.mkdir(parents=True)

        return directory


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class MicroworldClassifier(GraphComponent, IntentClassifier):
    """
    DIET-based classifier that detects the microworld-specific intent and entities.
    """

    microworlds: List[str] = get_microworlds()

    def __init__(
            self,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext
    ) -> None:
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

        self.classifiers: Dict[str, DIETClassifier] = {}

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext
    ) -> GraphComponent:
        instance = cls(config, model_storage, resource, execution_context)

        for microworld in cls.microworlds:
            mw_config = DIETClassifier.get_default_config()

            # Determine configuration for the microworld that is being set up
            if 'microworlds' in config:
                mw_cfg = config['microworlds']
                assert microworld in mw_cfg
                mw_config.update(mw_cfg[microworld].copy())
            else:
                mw_config.update(config)

            # Create classifier for the specific microworld
            mw_output_fingerprint = f"{resource.output_fingerprint}_{microworld}"
            mw_model_storage = MultiModelStorage(model_storage._storage_path, microworld)
            mw_resource = Resource(resource.name, output_fingerprint=mw_output_fingerprint)
            instance.classifiers[microworld] = DIETClassifier.create(mw_config,
                                                                     mw_model_storage,
                                                                     mw_resource,
                                                                     execution_context)

        return instance

    @classmethod
    def required_components(cls) -> List[Type]:
        return [Featurizer]

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return DIETClassifier.required_packages()

    def process(self, messages: List[Message]) -> List[Message]:
        """
        Proxy component that dispatches the classification task to the classifier specific
        to the microworld the current message belongs to.

        :param messages: List of :class:`rasa.shared.nlu.training_data.message.Message` items to process.
        :return The list of processed messages
        """

        for message in messages:
            # Select the classifier corresponding to the detected microworld
            generic_intent = message.get(INTENT).get(INTENT_NAME_KEY)

            # Rename the intent information extracted by the domain classifier
            message.set(GENERIC_INTENT, message.get(INTENT), add_to_output=True)
            message.set(GENERIC_INTENT_RANKING, message.get(INTENT_RANKING_KEY), add_to_output=True)

            # Call the classifier, passing the arguments unchanged
            self.classifiers[generic_intent.split('.')[0]].process([message])

        return messages

    def train(self, training_data: TrainingData) -> Resource:
        """Train the embedding intent classifier on a data set."""

        for microworld in self.microworlds:
            # Extract the training examples corresponding to the current microworld
            mw_training_data = training_data.filter_training_examples(
                lambda msg: 'intent' not in msg.data or
                            msg.data['intent'].startswith(microworld)
            )

            logger.info(f"Training DIETClassifier for microworld: {microworld}. "
                        f"Number of examples: {len(mw_training_data.intent_examples)}")
            self.classifiers[microworld].train(mw_training_data)

        return self._resource

    def persist(self) -> None:
        for microworld in self.microworlds:
            # Each microworld model will be persisted separately according to its _resource member
            self.classifiers[microworld].persist()

    @classmethod
    def load(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            **kwargs: Any,
    ) -> "MicroworldClassifier":
        """Loads the trained model from the provided directory."""

        mw_classif = cls(config, model_storage, resource, execution_context)

        for microworld in cls.microworlds:
            mw_config = DIETClassifier.get_default_config()
            mw_config.update(config)

            mw_output_fingerprint = f"{resource.output_fingerprint}_{microworld}"
            mw_resource = Resource(resource.name, output_fingerprint=mw_output_fingerprint)
            mw_model_storage = MultiModelStorage(model_storage._storage_path, microworld)
            mw_classif.classifiers[microworld] = DIETClassifier.load(mw_config,
                                                                     mw_model_storage,
                                                                     mw_resource,
                                                                     execution_context,
                                                                     **kwargs)

        return mw_classif
