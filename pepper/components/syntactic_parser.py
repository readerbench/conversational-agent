from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import (
    TEXT
)

import spacy
from spacy.tokens import Span
from spacy.lookups import Lookups
import spacy.symbols as symbols
from spacy.symbols import POS, ADJ, ADP, ADV, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SYM, VERB, X, CCONJ, SCONJ, DET, AUX
from spacy.lang.ro.tag_map import TAG_MAP

SEMANTIC_ROLES = "semantic_roles"


class SyntacticParser(Component):
    """ Component that identifies syntactic entities of the phrase by their syntactic question. """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """ Specify which components need to be present in the pipeline. """

        return []

    name = "SyntacticParser"

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = ['ro']

    @staticmethod
    def __load_lemmas():
        # Extract lemma lookup tables from the compressed binary file (.bin)
        lookups = spacy.lookups.Lookups()
        lookups.from_disk('./microworlds/mem_assistant/data/lookups')
        noun_lemmas = lookups.get_table("noun-lemmas")
        prop_noun_lemmas = lookups.get_table("prop-noun-lemmas")
        verb_lemmas = lookups.get_table("verb-lemmas")

        # Manually create a pronoun lemma lookup table
        pronouns = {'meu': 'eu', 'mea': 'eu', 'mei': 'eu', 'mele': 'eu',
                    'lui': 'el', 'lor': 'ei'}
        pron_lemmas = spacy.lookups.Table.from_dict(pronouns)

        return {
            symbols.NOUN: noun_lemmas,
            symbols.PROPN: prop_noun_lemmas,
            symbols.VERB: verb_lemmas,
            symbols.PRON: pron_lemmas,
        }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

        # Initialize spaCy model used for syntactic-semantic parsing
        self.nlp_spacy = spacy.load('./models/spacy-syntactic')

        # Initialize the lemmatizer
        self.lemmas = SyntacticParser.__load_lemmas()

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided with the training data. The component can rely on
        any context attribute to be present, that gets created by a call to :meth:`components.Component.pipeline_init`
        of ANY component and on any context attributes created by a call to :meth:`components.Component.train`
        of components previous to this one."""
        pass

    @staticmethod
    def __part_of_speech_from_tag(tag):
        """ Return the part of speech code for a given POS detailed tag. """

        stub_tag_map = {
            "A": symbols.ADJ,
            "D": symbols.DET,
            "C": symbols.CCONJ,
            "I": symbols.INTJ,
            "M": symbols.NUM,
            "N": symbols.NOUN,
            "P": symbols.PRON,
            "R": symbols.ADV,
            "T": symbols.DET,
            "V": symbols.VERB,
        }

        if tag in TAG_MAP:
            return TAG_MAP[tag][symbols.POS]
        return stub_tag_map.get(tag[0])

    def __lemmatize(self, token):
        """
        Get the lemma for the given inflected word.

        :return the word's lemma ot the same word if the lemma is not available in the lookup tables
        """

        tag = token.tag_.split('__')[0]  # Extract the compact tag
        pos = SyntacticParser.__part_of_speech_from_tag(tag)
        word = token.text

        # POS = proper noun
        if word in self.lemmas[symbols.PROPN]:
            return self.lemmas[symbols.PROPN][word]

        # POS = pronoun
        if word in self.lemmas[symbols.PRON]:
            return self.lemmas[symbols.PRON][word]

        # POS = noun
        if pos == symbols.NOUN:
            return self.lemmas[symbols.NOUN].get(word, word)

        # POS = verb
        if pos == symbols.VERB:
            return self.lemmas[symbols.VERB].get(word, word)

        # TODO numeral - doi/două -> 2

        return word

    def __get_dependency_span(self, doc, token, include_all_deps=False):
        """
        Build the span of words connected to a given token
        (representing attributes/prepositions/articles etc).
        """

        def dfs(node):
            """
            Depth First Search through the dependency tree
            to determine the span that includes a certain token.
            """

            first = last = node.i
            prep_first = prep_last = None
            for child in node.children:
                if child.dep_ in ['-', 'prep', 'cât'] or \
                        (include_all_deps and child.dep_ in ['care', 'ce fel de', 'al cui']):
                    child_first, child_last, _, _ = dfs(child)

                    if child.dep_ == 'prep':
                        prep_first = child_first
                        prep_last = child_last
                    else:
                        first = min(first, child_first)
                        last = max(last, child_last)

            return first, last, prep_first, prep_last

        first, last, prep_first, prep_last = dfs(token)  # compute bounds of the span
        span = Span(doc, first, last + 1)

        prep_span = Span(doc, prep_first, prep_last + 1) if prep_first is not None else None
        prep = prep_span.text if prep_span else ""

        return span.text, prep

    def __get_specifiers(self, doc, parent):
        """ Extract attributes (that identifies a specific instance of an entity) of a given token. """

        specifiers = []
        for token in doc:
            if token.head == parent:
                if token.dep_ in ['care', 'ce fel de', 'cât'] or \
                        token.dep_ == 'cât timp' and token.head.dep_ != "ROOT":
                    ext_value, prep = self.__get_dependency_span(doc, token, True)
                    specifiers.append({
                        "question": token.dep_,
                        "determiner": self.__get_dependency_span(doc, token.head)[0],
                        "pre": prep,
                        "value": ext_value,
                        "lemma": self.__get_dependency_span(doc, token, True)[0],
                        "specifiers": []
                    })
                elif token.dep_ in ['al cui']:
                    specifiers.append({
                        "question": token.dep_,
                        "determiner": self.__get_dependency_span(doc, token.head)[0],
                        "value": token.text,
                        "lemma": self.__lemmatize(token),
                        "specifiers": self.__get_specifiers(doc, token)
                    })

        return specifiers

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming message. The component can rely on any context
        attribute to be present, that gets created by a call to :meth:`components.Component.pipeline_init`
        of ANY component and on any context attributes created by a call to :meth:`components.Component.process`
        of components previous to this one."""

        # Parse the phrase
        text = message.get(TEXT)
        if not text:
            return
        doc = self.nlp_spacy(text.lower())

        semantic_roles = []
        inferred_subj = None

        for token in doc:
            # Identify principal components of the sentence
            if token.dep_ not in ['-'] and token.head.dep_ == 'ROOT':
                ext_value, prep = self.__get_dependency_span(doc, token, True)
                semantic_roles.append({
                    "question": token.dep_,
                    "determiner": self.__get_dependency_span(doc, token.head)[0],
                    "pre": prep,
                    "value": token.text,
                    "lemma": self.__lemmatize(token),
                    "ext_value": ext_value,
                    "specifiers": self.__get_specifiers(doc, token)
                })

            # Infer the subject (me) if the action is at the 1st person, singular
            if token.dep_ == 'ROOT' and (
                    (TAG_MAP[token.tag_.split('__')[0]].get('Person', '') == '1' and
                     TAG_MAP[token.tag_.split('__')[0]].get('Number', '') == 'Sing')
                    or
                    any(t.text == 'am' and t.dep_ == '-' and t.head.dep_ == 'ROOT' for t in doc)
            ):
                inferred_subj = {
                    "question": "cine",
                    "determiner": self.__get_dependency_span(doc, token)[0],
                    "value": "eu",
                    "lemma": "eu",
                    "ext_value": "eu",
                    "specifiers": []
                }

        if inferred_subj and not any(ent['question'] == 'cine' for ent in semantic_roles):
            semantic_roles.append(inferred_subj)

        # Add extracted entities to the message
        message.set(SEMANTIC_ROLES, semantic_roles, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""

        pass
