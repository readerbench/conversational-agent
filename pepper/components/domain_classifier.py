from typing import Any, Optional

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import INTENT

from copy import deepcopy


class DomainClassifier(DIETClassifier):
    """
    DIET-based Classifier for detecting the microworld (the generic domain) that an utterance represents.
    """

    def train(self, training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any) -> None:

        # Preprocess the training data: Alter the intents of the train examples to their domain (microworld)
        domain_train_data = deepcopy(training_data)
        for ex in domain_train_data.training_examples:
            if ex.get(INTENT):
                ex.set(INTENT, ex.get(INTENT).split('.')[0])

        # Use default training method
        super().train(domain_train_data, config, **kwargs)
