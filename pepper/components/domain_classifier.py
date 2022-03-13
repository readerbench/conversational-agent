from copy import deepcopy

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import INTENT


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class DomainClassifier(DIETClassifier):
    """
    DIET-based Classifier for detecting the microworld (the generic domain) that an utterance represents.
    """

    def train(self, training_data: TrainingData) -> Resource:

        # Preprocess the training data: Alter the intents of the train examples to their domain (microworld)
        domain_train_data = deepcopy(training_data)
        for ex in domain_train_data.training_examples:
            if ex.get(INTENT):
                ex.set(INTENT, ex.get(INTENT).split('.')[0])

        # Use default training method
        return super().train(domain_train_data)
