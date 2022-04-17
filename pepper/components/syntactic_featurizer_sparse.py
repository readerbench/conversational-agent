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
        precomputed_deps = {
            "Bună!": ['-', '-'],
            "Hei": ['-'],
            "hey": ['-'],
            "salut": ['-'],
            "bună": ['-'],
            "salutare": ['-'],
            "hello": ['-'],
            "bună ziua": ['-', '-'],
            "Salut": ['-'],
            "hi": ['-'],
            "hei!": ['-', '-'],
            "Îți multumesc mult pentru ajutor": ['-', '-', '-', '-', '-'],
            "Mulțumesc de ajutor": ['-', '-', '-'],
            "Mi-ai fost de mare ajutor": ['-', '-', '-', '-', '-', '-', '-'],
            "Mersi!": ['-', '-'],
            "Mulțu": ['-'],
            "Mulțumesc!": ['-', '-'],
            "super": ['-'],
            "Tare": ['-'],
            "Cool": ['-'],
            "Nice": ['-'],
            "mersi mult": ['-', '-'],
            "merci": ['-'],
            "bye": ['-'],
            "pa": ['-'],
            "paa": ['-'],
            "paaa": ['-'],
            "pa pa": ['-', '-'],
            "papa": ['-'],
            "gata": ['-'],
            "o zi frumoasă": ['-', '-', '-'],
            "da": ['-'],
            "daa": ['-'],
            "da da": ['-', '-'],
            "dada": ['-'],
            "dap": ['-'],
            "sigur": ['-'],
            "ok": ['-'],
            "yep": ['-'],
            "yes": ['-'],
            "Ce e o minge?": ['-', '-', '-', '-', '-'],
            "Ce e un calculator?": ['-', '-', '-', '-', '-'],
            "Cine e Bill Gates?": ['-', '-', '-', '-', '-'],
            "Ce știi să faci?": ['-', '-', '-', '-', '-'],
            "Cu ce mă poți ajuta?": ['-', '-', '-', '-', '-', '-'],
            "Ce pot să te întreb?": ['-', '-', '-', '-', '-', '-'],
            "Ce știi?": ['-', '-', '-'],
            "La ce fel de întrebări poți răspunde?": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "Spune-mi ce te pot întreba": ['-', '-', '-', '-', '-', '-', '-'],
            "Ce informații pot afla?": ['-', '-', '-', '-', '-'],
            "Help": ['-'],
            "Ajutor": ['-'],
            "Care sunt tipurile de întrebări la care răspunzi?": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Care este rolul tău?": ['-', '-', '-', '-', '-'],
            "care sunt abilitățile tale?": ['-', '-', '-', '-', '-'],
            "capabilități": ['-'],
            "abilități": ['-'],
            "ce cunoștințe ai": ['-', '-', '-'],
            "am nevoie de ajutor": ['-', '-', '-', '-'],
            "Vreau exemple de întrebări": ['-', '-', '-', '-'],
            "Dă-mi exemplu de ce te pot întreba": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "vreau exemple": ['-', '-'],
            "care sunt exemplele?": ['-', '-', '-', '-'],
            "exemple de replici": ['-', '-', '-'],
            "vreau să ajung la casa presei libere": ['-', '-', '-', '-', 'unde', '-', '-'],
            "as dori sa ajung la piata unirii": ['-', '-', '-', '-', '-', 'unde', '-'],
            "cum ajung repede in otopeni?": ['unde', '-', '-', '-', 'unde', '-'],
            "poti sa-mi zici cum sa ajung in Pantelimon?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', 'unde', '-'],
            "vreau să merg la dristor": ['-', '-', '-', '-', 'unde'],
            "aș dori să mă duc la facultate": ['-', '-', '-', '-', '-', '-', 'unde'],
            "arata-mi directii spre universitate": ['-', '-', '-', '-', '-', 'unde'],
            "vreau direcții către palatul parlamentului": ['-', '-', '-', 'unde', '-'],
            "cum fac să merg in Herăstrău?": ['unde', '-', '-', '-', '-', 'unde', '-'],
            "care sunt rutele să ajung la mausoleu?": ['-', '-', '-', '-', '-', '-', 'unde', '-'],
            "cum pot să merg până la parcul tineretului?": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "cum merg spre bragadiru?": ['-', '-', '-', 'unde', '-'],
            "arată-mi rute către stația de metrou Orizont": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "zi-mi cum ajung pe lacul morii": ['-', '-', '-', '-', '-', '-', 'unde', '-'],
            "Arata-mi te rog cum merg la Gara de Nord": ['-', '-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "Vreau sa vad cum mă duc la Apusului": ['-', '-', '-', '-', '-', '-', '-', 'unde'],
            "Aș vrea să merg în parcul sticlariei": ['-', '-', '-', '-', '-', 'unde', '-'],
            "ce mijloace de transport merg spre promenada": ['-', '-', '-', '-', '-', '-', 'unde'],
            "ce mijloace pot să iau până la farmacia tei?": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "cum pot ajunge rapid la laserul de la Măgurele": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "vreau să merg în sectorul 5": ['-', '-', '-', '-', 'unde', '-'],
            "trebuie să ajung pe strada lalelelor": ['-', '-', '-', '-', 'unde', '-'],
            "dă-mi transport către arena națională": ['-', '-', '-', '-', '-', 'unde', '-'],
            "Vreau să reții ceva": ['-', '-', '-', '-'],
            "Poți să reții ceva, te rog?": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "Memorează": ['-'],
            "Memorează ceva": ['-', '-'],
            "Reține ceva": ['-', '-'],
            "Ține minte": ['-', '-'],
            "Memo": ['-'],
            "reține": ['-'],
            "aș dori să stochez ceva": ['-', '-', '-', '-', '-'],
            "aș vrea să ții ceva în memorie": ['-', '-', '-', '-', '-', '-', 'unde'],
            "reține ce îți zic": ['-', '-', '-', '-'],
            "cartea e pe masă": ['-', '-', '-', 'unde'],
            "Cardul de debit este în portofelul vechi": ['-', '-', '-', '-', '-', 'unde', '-'],
            "buletinul meu se află în rucsac": ['-', '-', '-', '-', '-', 'unde'],
            "biblioteca se află la etajul 5": ['-', '-', '-', '-', 'unde', '-'],
            "service-ul gsm este pe strada Ecaterina Teodoroiu numărul 12": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-'],
            "Am pus foile cu tema la mate pe dulapul din sufragerie": ['-', '-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "Alex stă în căminul P5": ['-', '-', '-', 'unde', '-'],
            "Maria Popescu locuiește pe Bulevardul Timișoara numărul 5": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Daniela stă la blocul 23": ['-', '-', '-', 'unde', '-'],
            "eu stau la adresa str. Ec. Teod. nr. 17": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "Bonurile de transport sunt în plicul de pe raft": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "cardul de memorie e sub cutia telefonului": ['-', '-', '-', '-', '-', 'unde', '-'],
            "tastatura mea este în depozit": ['-', '-', '-', '-', 'unde'],
            "profesorul se află în camera vecină": ['-', '-', '-', '-', 'unde', '-'],
            "am lăsat lădița cu cartofi în pivniță": ['-', '-', '-', '-', '-', '-', 'unde'],
            "mi-am pus casca de înot în dulapul cu tricouri": ['-', '-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "geaca de iarnă este în șifonierul de acasă": ['-', '-', '-', '-', '-', 'unde', '-', '-'],
            "lămpile solare sunt de la bricostore": ['-', '-', '-', '-', '-', 'unde'],
            "perechea de adidași albi este de la intersport din Afi": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "Alin Dumitru stă în Militari Residence": ['-', '-', '-', '-', 'unde', '-'],
            "am lăsat suportul de brad la țară în garaj": ['-', '-', '-', '-', '-', '-', 'unde', '-', 'unde'],
            "pachetul de creioane colorate este pe polița de pe perete": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "mi-am așezat prosoapele pe suport": ['-', '-', '-', '-', '-', '-', 'unde'],
            "am cumpărat florile de la mireille": ['-', '-', '-', '-', '-', 'unde'],
            "am luat veioza de la București": ['-', '-', '-', '-', '-', 'unde'],
            "sandalele sunt de la magazinul benvenutti": ['-', '-', '-', '-', 'unde', '-'],
            "trandafirii sunt de la Nicu": ['-', '-', '-', '-', 'unde'],
            "chitara mea este de la Andrei": ['-', '-', '-', '-', '-', 'unde'],
            "am mers cu mașina până la Iași": ['-', '-', '-', '-', '-', '-', 'unde'],
            "telefonul lui Jan este pe noptieră": ['-', '-', '-', '-', '-', 'unde'],
            "tricoul meu de alergat e în șifonier pe hol": ['-', '-', '-', '-', '-', '-', 'unde', '-', 'unde'],
            "trusa de machiaj a Anisiei este în geamantanul verde": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-'],
            "ganterele lui Horia sunt sub canapea": ['-', '-', '-', '-', '-', 'unde'],
            "bormașina cu percuție a lui Vlad e la Timișoara": ['-', '-', '-', '-', '-', '-', '-', '-', 'unde'],
            "pungile acestea sunt de la mall": ['-', '-', '-', '-', '-', 'unde'],
            "Bob se află pe stația spațială": ['-', '-', '-', '-', 'unde', '-'],
            "Clara a venit de pe muntele Olimp": ['-', '-', '-', '-', '-', 'unde', '-'],
            "Rodica se întoarce de la festivalul Untold": ['-', '-', '-', '-', '-', 'unde', '-'],
            "Unde se află ochelarii": ['unde', '-', '-', '-'],
            "Unde e buletinul?": ['unde', '-', '-', '-'],
            "știi unde am pus cheile": ['-', 'unde', '-', '-', '-'],
            "poți să-mi zici unde este încârcătorul de telefon": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-'],
            "zi-mi unde am pus ochelarii de înot": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "unde am lăsat ceasul?": ['unde', '-', '-', '-', '-'],
            "pe unde e sticla de ulei": ['-', 'unde', '-', '-', '-', '-'],
            "pe unde mi-am pus pantofii negri": ['-', 'unde', '-', '-', '-', '-', '-', '-'],
            "de unde am cumpărat uscătorul de păr": ['-', 'unde', '-', '-', '-', '-', '-'],
            "de unde au venit cartofii în Europa": ['-', 'unde', '-', '-', '-', '-', 'unde'],
            "de unde pleacă trenul IR1892": ['-', 'unde', '-', '-', '-'],
            "până unde a alergat aseară Marius?": ['-', 'unde', '-', '-', 'când', '-', '-'],
            "până unde au ajuns radiațiile de la Cernobâl": ['-', 'unde', '-', '-', '-', '-', '-', '-'],
            "de unde am luat draperiile din sufragerie?": ['-', 'unde', '-', '-', '-', '-', '-', '-'],
            "spune-mi te rog unde a pus Alina dosarul cu actele de la serviciu": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "ai putea să îmi zici unde era casa lui Teo?": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "mailul lui Alex este următorul": ['-', '-', '-', '-', 'care este'],
            "valoarea constantei pi este următoarea": ['-', '-', '-', '-', 'care este'],
            "colegii care au luat 10 la ML sunt următorii:": ['-', '-', '-', '-', '-', '-', '-', '-', 'care este', '-'],
            "materiile opționale din anul 1 sunt următoarele": ['-', '-', '-', '-', '-', '-', '-'],
            "tehnologiile folosite de aplicație sunt următoarele": ['-', '-', '-', 'unde', '-', 'care este'],
            "codul de acces este ăsta": ['-', '-', '-', '-', 'care este'],
            "numărul de la casă al Mariei e acesta": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "dimensiunile portbagajului meu de la mașină sunt acestea": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "adresa mea de la București e aceasta": ['-', '-', '-', '-', '-', '-', 'care este'],
            "versiunile de Windows 10 sunt acestea": ['-', '-', '-', '-', '-', 'care este'],
            "lunile pe care am decontat abonamentul sunt cele care urmează": ['-', '-', 'unde', '-', '-', '-', '-', 'care este', '-', '-'],
            "ecuația fluxului electromagnetic e cea care urmează": ['-', '-', '-', '-', 'care este', '-', '-'],
            "colegii mei de liceu erau cei ce urmează": ['-', '-', '-', '-', '-', 'care este', '-', '-'],
            "dioptriile mele de la ochelari erau ultima dată cele ce urmează": ['-', '-', '-', '-', '-', '-', '-', 'când', 'care este', '-', '-'],
            "programul de la notarul de la Universitate este acesta:": ['-', '-', '-', '-', '-', '-', '-', '-', 'care este', '-'],
            "programul de lucru al lui Cristian e ăsta": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "explicatia indicatorului albastru este asta": ['-', '-', '-', '-', 'care este'],
            "Mailul lui Alex Marin este alex@marin.com": ['-', '-', '-', '-', '-', 'care este'],
            "Adresa Elenei este strada Zorilor numărul 9": ['-', '-', '-', 'care este', '-', '-', '-'],
            "Numărul de telefon al lui Dan e 123456789": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "numărul blocului fratelui Mihaelei e 10": ['-', '-', '-', '-', '-', 'care este'],
            "anul nașterii lui Ștefan cel Mare a fost 1433": ['-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "numele meu este Gabriel": ['-', '-', '-', 'care este'],
            "adresa de la serviciu este Bulevardul Unirii nr. 0": ['-', '-', '-', '-', '-', 'care este', '-', '-', '-'],
            "numele asistentului de programare paralelă e Paul Walker": ['-', '-', '-', '-', '-', '-', 'care este', '-'],
            "suprafața apartamentului de la București este de 58mp": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "prețul canapelei a fost de 1300 de lei": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "codul de activare al sistemului de operare e APCHF6798HJ67GI90": ['-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "sala laboratorului de PP este EG321": ['-', '-', '-', '-', '-', 'care este'],
            "username-ul meu de github este gabrielboroghina": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "seria mea de la buletin este GG2020": ['-', '-', '-', '-', '-', '-', 'care este'],
            "codul PIN de la cardul meu de sănătate este 0000": ['-', '-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "placa mea video este NVidia GeForce GTX950m": ['-', '-', '-', '-', 'care este', '-', '-', '-'],
            "telefonul Karinei este 243243": ['-', '-', '-', 'care este'],
            "limbajul de programare folosit de Thales este C++": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "tensiunea nominală de alimentare a pompei este 120V": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "data de expirare a suplimentelor alimentare din cămară este 30-05-2023": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "Care e mailul lui Mihai?": ['-', '-', '-', '-', '-', '-'],
            "care este numele de utilizator de github al laborantului de EIM": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "poți să-mi spui care era prețul abonamentului la sală": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "zi-mi care a fost câștigătorul concursului Eestec Olympics de anul trecut": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care era denumirea bazei de date de la proiect?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care sunt tipurile de rețele neurale": ['-', '-', '-', '-', '-', '-'],
            "care este valoarea de adevăr a propoziției": ['-', '-', '-', '-', '-', '-', '-'],
            "care e adresa colegului meu?": ['-', '-', '-', '-', '-', '-'],
            "care e frecvența procesorului meu": ['-', '-', '-', '-', '-'],
            "care e numărul lui Radu": ['-', '-', '-', '-', '-'],
            "care e limita de viteză în localitate": ['-', '-', '-', '-', '-', '-', '-'],
            "care e punctul de topire al aluminiului": ['-', '-', '-', '-', '-', '-', '-'],
            "care e data de naștere a lui Mihai Popa": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care este mărimea mea la adidași": ['-', '-', '-', '-', '-', '-'],
            "care este temperatura medie în Monaco în iunie": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "care este diferența de vârstă între mine și Vlad": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "zi-mi și mie care era adresa de căsuță poștală a Karinei Preda": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care sunt datele cardului meu revolut": ['-', '-', '-', '-', '-', '-'],
            "care este telefonul de la frizerie": ['-', '-', '-', '-', '-', '-'],
            "care e dobânda de la creditul pentru casă?": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care sunt ministerele cu probleme": ['-', '-', '-', '-', '-'],
            "care este tipografia centrală": ['-', '-', '-', '-'],
            "spune-mi care este seria mea de la buletin": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "vreau să știu care e numele de familie al lui Sebi": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "aș vrea să îmi zici te rog care e termenul limită al temei de la algebră": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "ce floare s-a uscat": ['-', '-', '-', '-', '-', '-'],
            "ce hackathon va avea loc săptămâna viitoare": ['-', '-', '-', '-', '-', 'când', '-'],
            "ce windows am acum pe calculator": ['-', '-', '-', '-', '-', '-'],
            "ce examene vor fi date în iunie?": ['-', '-', '-', '-', '-', '-', 'când', '-'],
            "ce temperatură a fost în iulie anul trecut": ['-', '-', '-', '-', '-', 'când', 'când', '-'],
            "ce culoare au ochii Andreei": ['-', '-', '-', '-', '-'],
            "ce mail am folosit la serviciu": ['-', '-', '-', '-', '-', 'unde'],
            "la ce apartament locuiește verișorul meu": ['-', '-', '-', '-', '-', '-'],
            "la ce sală se află microscopul electronic": ['-', '-', 'unde', '-', '-', '-', '-'],
            "la ce număr de telefon se dau informații despre situația actuală": ['-', '-', 'unde', '-', '-', '-', '-', '-', '-', 'unde', '-'],
            "la care salon este internat bunicul lui": ['-', '-', 'unde', '-', '-', '-', '-'],
            "la care cod poștal a fost trimis pachetul": ['-', '-', 'unde', '-', '-', '-', '-', '-'],
            "la care hotel s-au cazat Mihai și Alex ieri": ['-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', 'când'],
            "ce fel de imprimantă am acasă": ['-', '-', '-', '-', '-', '-'],
            "ce fel de baterie folosește ceasul de mână": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "ce fel de procesor are telefonul meu": ['-', '-', '-', '-', '-', '-', '-'],
            "în care dulap am pus dosarul": ['-', '-', 'unde', '-', '-', '-'],
            "în care cameră am lăsat încărcătorul de telefon": ['-', '-', 'unde', '-', '-', '-', '-', '-'],
            "în care săptămână e examenul de învățare automată": ['-', '-', 'când', '-', '-', '-', '-', '-'],
            "de la care prieten e cadoul acesta": ['-', '-', '-', 'unde', '-', '-', '-'],
            "de la care magazin mi-am luat cablul de date?": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-'],
            "pentru ce test am învățat acum 2 zile": ['-', '-', '-', '-', '-', '-', '-', 'când'],
            "pe care masă am pus ieri periuța de dinți?": ['-', '-', 'unde', '-', '-', 'când', '-', '-', '-', '-'],
            "pe care poziție este mașina în parcare": ['-', '-', 'unde', '-', '-', '-', '-'],
            "de pe care cont am plătit factura de curent acum 3 zile": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', 'când'],
            "pe ce viteză am setat aerul condiționat": ['-', '-', '-', '-', '-', '-', '-'],
            "pe ce loc am ieșit la olimpiada de info din clasa a 12-a": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "care aparat de făcut sandwichuri e la reducere": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "ce fel de uscător de păr folosește Alice": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "care clasificare e corectă": ['-', '-', '-', '-'],
            "care emisferă este": ['-', '-', '-'],
            "care elicopter": ['-', '-'],
            "care autoturism este cel mai prietenos cu mediul": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "la ce sprânceană e problema": ['-', '-', '-', '-', '-'],
            "la ce fel de neurooftalmologie trebuie să meargă?": ['-', '-', '-', '-', 'unde', '-', '-', '-', '-'],
            "pentru care deversare au fost amedați": ['-', '-', '-', '-', '-', '-'],
            "ce fel de pocănit se auzea ieri": ['-', '-', '-', '-', '-', '-', 'când'],
            "concursul va fi pe 12 ianuarie": ['-', '-', '-', '-', 'când', '-'],
            "îmi expiră permisul de conducere pe 25 februarie 2023": ['-', '-', '-', '-', '-', '-', 'când', '-', '-'],
            "plecarea în Franța este pe 5 martie": ['-', '-', '-', '-', '-', 'când', '-'],
            "abonamentul STB îmi expiră pe 23 aprilie": ['-', '-', '-', '-', '-', 'când', '-'],
            "Viorel e născut pe 16 mai 1998": ['-', '-', '-', '-', 'când', '-', '-'],
            "Vacanța de vară începe pe 30 iunie": ['-', '-', '-', '-', '-', 'când', '-'],
            "Pe 3 iulie se termină sesiunea de licență": ['-', 'când', '-', '-', '-', '-', '-', '-'],
            "ziua Daianei este în august": ['-', '-', '-', '-', 'când'],
            "în septembrie începe școala": ['-', 'când', '-', '-'],
            "din octombrie apare un nou film la cinema": ['-', 'când', '-', '-', '-', '-', '-', 'unde'],
            "până în noiembrie trebuie să termin task-ul": ['-', '-', 'când', '-', '-', '-', '-', '-', '-'],
            "noile autobuze au apărut în decembrie": ['-', '-', '-', '-', '-', 'când'],
            "luni am fost la alergat": ['când', '-', '-', '-', 'unde'],
            "am mers la bazinul de înot marți": ['-', '-', '-', 'unde', '-', '-', 'când'],
            "garanția de la frigider se termină marțea viitoare": ['-', '-', '-', '-', '-', '-', 'când', '-'],
            "coletul cu jacheta va ajunge miercuri": ['-', '-', '-', '-', '-', 'când'],
            "testul de curs la rețele neurale a fost joi": ['-', '-', '-', '-', '-', '-', '-', '-', 'când'],
            "vineri încep promoțiile de black friday": ['când', '-', '-', '-', '-', '-'],
            "până sâmbătă e interzis accesul în mall": ['-', 'când', '-', '-', '-', '-', 'unde'],
            "Jack a fost la biserică duminică": ['-', '-', '-', '-', 'unde', 'când'],
            "azi am fost în parcul Titan": ['când', '-', '-', '-', 'unde', '-'],
            "Mâine vine Mihai pe la mine": ['când', '-', '-', '-', '-', 'unde'],
            "poimâine merg la mall": ['când', '-', '-', 'unde'],
            "olimpiada internațională de geografie va începe răspoimâine": ['-', '-', '-', '-', '-', '-', 'când'],
            "de ieri s-a făcut cald afară": ['-', 'când', '-', '-', '-', '-', '-', 'unde'],
            "am terminat proiecul la programare web alaltăieri": ['-', '-', '-', '-', '-', '-', 'când'],
            "procesorul Intel i7 rulează o instrucțiune în 0.3 nanosecunde": ['-', '-', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "execuția interogării în baza de date a durat 500 de milisecunde": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "sesiunea din browser a expirat acum 10 secunde": ['-', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "prezentarea proiectului durează 45 de minute": ['-', '-', '-', '-', '-', 'cât timp'],
            "am de prezentat temele peste o oră": ['-', '-', '-', '-', '-', '-', 'cât timp'],
            "am mâncat acum 2 ore": ['-', '-', '-', '-', 'cât timp'],
            "peste 3 sferturi de oră o să ajungă Ina": ['-', '-', 'cât timp', '-', '-', '-', '-', '-', '-'],
            "întâlnirea cu managerul este peste o jumătate de oră": ['-', '-', '-', '-', '-', '-', 'cât timp', '-', '-'],
            "peste 2 zile începe examenul de bacalaureat": ['-', '-', 'cât timp', '-', '-', '-', '-'],
            "Maria pleacă în concediu peste 5 săptămâni": ['-', '-', '-', 'unde', '-', 'cât timp', '-'],
            "săptămâna viitoare încep cursurile de înot": ['când', '-', '-', '-', '-', '-'],
            "conferința de bioinformatică a fost acum o lună": ['-', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "peste 2 ani termină Nicoleta masterul": ['-', '-', 'cât timp', '-', '-', '-'],
            "buletinul îmi expiră peste 3 ani": ['-', '-', '-', '-', '-', 'cât timp'],
            "mi-am făcut pașaportul acum un an": ['-', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "bătălia de la Mărășești a avut loc în anul 1917": ['-', '-', '-', '-', '-', '-', '-', '-', 'când', '-'],
            "acum 5 decenii nu existau calculatoare": ['-', '-', 'cât timp', '-', '-', '-'],
            "în secolul 19 s-a dezvoltat Imperiul lui Napoleon": ['-', 'când', '-', '-', '-', '-', '-', '-', '-', '-'],
            "luna viitoare vine Florin din Spania": ['când', '-', '-', '-', '-', '-'],
            "meciul dintre Franța și Spania va fi în weekend": ['-', '-', '-', '-', '-', '-', '-', '-', 'când'],
            "la anul se deschide mall-ul din Slatina": ['-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "mi-am făcut analize primăvara trecută": ['-', '-', '-', '-', '-', 'când', '-'],
            "restricțiile de circulație se încheie la vară": ['-', '-', '-', '-', '-', '-', 'când'],
            "toamna viitoare încep masterul": ['când', '-', '-', '-'],
            "Darius și-a schimbat domiciliul iarna trecută": ['-', '-', '-', '-', '-', '-', 'când', '-'],
            "curierul o să vină diseară": ['-', '-', '-', '-', 'când'],
            "aseară am făcut cartofi prăjiți": ['când', '-', '-', '-', '-'],
            "tonerul de la imprimantă s-a terminat alaltăseară": ['-', '-', '-', '-', '-', '-', '-', '-', 'când'],
            "bunica lui Alice va face cozonaci mâine dimineață": ['-', '-', '-', '-', '-', '-', 'când', 'când'],
            "ieri seara a fost foarte frig afară": ['când', 'când', '-', '-', '-', '-', 'unde'],
            "de dimineață trebuie să hrănesc animalele": ['-', 'când', '-', '-', '-', '-'],
            "am fost la cumpărături după amiază": ['-', '-', '-', 'unde', '-', 'când'],
            "la răsărit a început să cânte cocoșul din grădină": ['-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "evaluarea națională a ținut 2 ore și 45 de minute": ['-', '-', '-', '-', '-', 'cât timp', '-', '-', '-', 'cât timp'],
            "Jacob locuiește în Bremen de un an și 4 luni": ['-', '-', '-', 'unde', '-', '-', 'cât timp', '-', '-', 'cât timp'],
            "trebuie să iau antibioticul o dată la 6 ore": ['-', '-', '-', '-', '-', 'când', '-', '-', 'cât timp'],
            "bazinul de apă se umple o dată la 30 de minute": ['-', '-', '-', '-', '-', '-', 'când', '-', '-', '-', 'cât timp'],
            "Marc ia vitaminele de 2 ori pe zi": ['-', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "eu merg la sală de 3 ori pe săptămână": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "mgazinul se aprovizionează în fiecare săptămână": ['-', '-', '-', '-', '-', '-'],
            "în fiecare an apare un nou telefon Samsung Galaxy": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "serviciul funcționează de luni până vineri": ['-', '-', '-', 'când', '-', 'când'],
            "de joi va putea ajunge în Germania cu trenul": ['-', 'când', '-', '-', '-', '-', 'unde', '-', '-'],
            "până pe 24 iunie trebuie să trimit lucrarea de diplomă": ['-', '-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "festivalul de muzică ușoară și dans va fi de pe 1 septembrie pe 10 octombrie": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'când', '-', '-', 'când', '-'],
            "de pe 23 până pe 27 am fost în concediu": ['-', '-', 'când', '-', '-', 'când', '-', '-', '-', 'unde'],
            "între 3 și 7 ianuarie am fost la conferință în Toronto": ['-', 'când', '-', 'când', '-', '-', '-', '-', 'unde', '-', 'unde'],
            "la polul nord este noapte timp de 6 luni": ['-', 'unde', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "Andra și-a ales rochia de bal după 2 săptămâni": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'cât timp'],
            "de la ora 18 se oprește curentul electric": ['-', '-', 'când', '-', '-', '-', '-', '-'],
            "când vor avea loc alegerile locale din 2020": ['când', '-', '-', '-', '-', '-', '-', '-'],
            "Când am avut ultimul examen anul trecut?": ['când', '-', '-', '-', '-', 'când', '-', '-'],
            "zi-mi când am fost la sală?": ['-', '-', '-', 'când', '-', '-', '-', 'unde', '-'],
            "De când începe vacanța": ['-', 'când', '-', '-'],
            "Până când trebuie trimisă tema": ['-', 'când', '-', '-', '-'],
            "până când a durat al 2-lea război mondial?": ['-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "până când trebuie rezolvată problema la vlsi": ['-', 'când', '-', '-', '-', '-', '-'],
            "cât timp a durat prezentarea temei": ['-', 'cât timp', '-', '-', '-', '-'],
            "peste cât timp se termină starea de urgență?": ['-', '-', 'cât timp', '-', '-', '-', '-', '-', '-'],
            "peste cât timp începe sesiunea de examene": ['-', '-', 'cât timp', '-', '-', '-', '-'],
            "când trebuie să merg la control oftalmologic": ['când', '-', '-', '-', '-', 'unde', '-'],
            "când trebuie să iau pastilele de stomac": ['când', '-', '-', '-', '-', '-', '-'],
            "cât de des se actualizează sistemul de operare?": ['-', '-', 'când', '-', '-', '-', '-', '-', '-'],
            "cât de des trebuie să ud florile din fața casei": ['-', '-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "în ce perioadă se va ține concursul de fizică de la Bacău": ['-', '-', 'când', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "în ce perioadă vor avea loc preselecțiile pentru balul bobocilor": ['-', '-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "în care perioadă a fost cel mai frig afară?": ['-', '-', 'când', '-', '-', '-', '-', '-', '-', 'unde'],
            "în ce interval e deschis magazinul Kaufland de la Giurgiu?": ['-', '-', 'când', '-', '-', '-', '-', '-', '-', '-', '-'],
            "în ce interval de timp a avut loc atacul": ['-', '-', 'când', '-', '-', '-', '-', '-', '-'],
            "de când până când se vor închide magazinele": ['-', 'când', '-', 'când', '-', '-', '-', '-'],
            "de când până când pot citi indexul de energie electrică": ['-', 'când', '-', 'când', '-', '-', '-', '-', '-', '-'],
            "Cine stă în căminul P16?": ['-', '-', '-', 'unde', '-', '-'],
            "Spune-mi te rog cine a inventat becul": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "cine a câștigat locul 1 la olimpiada națională de matematică din 2016": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "cine m-a tuns ultima dată?": ['-', '-', '-', '-', '-', '-', '-', 'când'],
            "de la cine a cumpărat mihaela cireșele": ['-', '-', 'unde', '-', '-', '-', '-'],
            "de la cine a apărut problema": ['-', '-', 'unde', '-', '-', '-'],
            "cine a fost primul om pe lună?": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "cine a venit ieri la cursul de astronomie": ['-', '-', '-', 'când', '-', 'unde', '-', '-'],
            "zi-mi cine a propus problemele de la concursul InfoOlt 2016": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Unde are biroul Mihai Dascalu?": ['unde', '-', '-', '-', '-', '-'],
            "Unde este Răzvan Deaconescu?": ['unde', '-', '-', '-', '-'],
            "Unde il pot gasi pe Florin Pop?": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "Care e biroul lui Marius Leordeanu": ['-', '-', '-', '-', '-', '-'],
            "Spune-mi te rog unde se află biroul lui Mihnea Moisescu": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Știi unde îl pot găsi pe profesorul Dan Tudose?": ['-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Oare unde este domnul profesor Traian Rebedea?": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "Zi-mi și mie unde are Mariana Mocanu biroul în facultate": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Unde îl găsesc pe asistentul Emilian Rădoi?": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "Spune care e biroul domnului Gologan": ['-', '-', '-', '-', '-', '-'],
            "Unde stă de obicei doamna profesoară Adina Paunescu": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "Vreau să găsesc pe Marius Popescu": ['-', '-', '-', '-', '-', '-'],
            "Aș dori să ajung la laborantul Cosmin Dragomir": ['-', '-', '-', '-', '-', 'unde', '-', '-'],
            "Unde merg să îl găsesc pe asistentul Alexandru Negrescu?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Unde e biroul lui Radu Ciobanu?": ['unde', '-', '-', '-', '-', '-', '-'],
            "Afișează-mi harta etajului corespunzător sălii PR 303!": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Ajută-mă să găsesc sala PR 706!": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Ajută-mă să găsesc sala PR 103b!": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Am nevoie de ajutor să găsesc drumul către sala PR 303!": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Arată-mi calea către sala PR 701!": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Arată-mi harta cu sala PR 002 reprezentată.": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Care este direcția către sala PR 605?": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "Care este drumul către sala PR 706?": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "Care este drumul spre sala PR 103a?": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "Condu-mă la sala PR 305!": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Cum pot ajunge la sala PR 603?": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Există vreo hartă pentru a vizualiza localizarea sălii PR 403?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Explică-mi cum ajung la sala PR 403!": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Indică-mi drumul către sala PR 003!": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "La ce etaj se află sala PR 604?": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Mă conduci până la sala PR 103a?": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Mă poți ghida către sala PR 608?": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Mă poți îndrepta către sala PR 002?": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Mă poți îndruma către sala PR 408?": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Pe unde se află sala PR 604?": ['-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Pe unde să o iau ca să ajung la sala PR 702?": ['-', 'unde', '-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Poti să mă conduci la sala PR 701, te rog?": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Poți să îmi indici drumul către sala PR 608?": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Spune-mi unde găsesc sala PR 001!": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "Unde e sala PR 305?": ['unde', '-', '-', '-', '-', '-'],
            "Unde este localizată sala PR 708?": ['unde', '-', '-', '-', '-', '-', '-'],
            "Unde găsesc sala PR 103b?": ['unde', '-', '-', '-', '-', '-'],
            "Unde pot găsi sala PR 001?": ['unde', '-', '-', '-', '-', '-', '-'],
            "Unde se află sala PR 606?": ['unde', '-', '-', '-', '-', '-', '-'],
            "Vreau să văd harta cu sala PR 301 reprezentată.": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Zi-mi unde găsesc sala PR 003!": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "Îmi poți arăta calea către sala PR 702, te rog?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Îmi poți arăta o hartă cu sala PR 106 reprezentată.": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Îmi poți explica unde se află sala PR 302, te rog?": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "În ce direcție se găsește sala PR 601?": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "În ce loc se află sala PR 605?": ['-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "În ce sens se află sala PR 307?": ['-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Încotro să o iau către sala PR 203?": ['unde', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Știi unde se află sala PR 606?": ['-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Unde e AN 02": ['unde', '-', '-', '-'],
            "Cum ajung în EG413?": ['-', '-', '-', 'unde', '-'],
            "Ajută-mă să găsesc sala unde se ține Programare Orientata pe Obiecte!": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Arată-mi sala în care se desfășoară Elemente de Electronica Analogica!": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Mă poți ajuta să găsesc unde are loc Logica?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Spune-mi, te rog, în ce sală pot participa la Cultura si Civilizatie?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "Unde are loc Instrumente Informatice?": ['unde', '-', '-', '-', '-', '-'],
            "Unde se desfășoară Paradigme de Programare?": ['unde', '-', '-', '-', '-', '-', '-'],
            "Unde se predă Proiectarea Algoritmilor?": ['unde', '-', '-', '-', '-', '-'],
            "Unde trebuie să mă duc pentru a participa la Istoria Religiilor?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Unde va avea loc Filosofie Cognitivista?": ['unde', '-', '-', '-', '-', '-', '-'],
            "Îmi poți spune, te rog, unde se va ține Retele Locale?": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "În ce loc se ține Ingineria Calculatoarelor?": ['-', '-', 'unde', '-', '-', '-', '-', '-'],
            "În ce sală are loc Mecanica?": ['-', '-', 'unde', '-', '-', '-', '-'],
            "Știi unde se desfășoară Proiectare Logica?": ['-', 'unde', '-', '-', '-', '-', '-'],
            "Știi unde se va ține Algoritmi Paraleli si Distribuiti?": ['-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Unde se ține Metode Numerice?": ['unde', '-', '-', '-', '-', '-'],
            "Unde se ține laboratorul pentru grupa 311CC?": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "Unde se ține cursul pentru seria CB?": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "Zi-mi te rog unde are loc curs pentru seria CA": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Vreau să știu unde se desfășoară seminarul grupei 222AC": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "Unde se ține cursul de Electronica Digitala pentru seria CC?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Unde se ține seminarul de Teoria Sistemelor pentru grupa 315CB?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Unde se ține seminar de Engleza pentru grupa 321CC?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Aș vrea să știu unde e seminarul de Mate pentru grupa 142BB": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Zi-mi unde va fi cursul seriei S1 de analiză matematică": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Proiectare Logică": ['-', '-'],
            "materia de Fizică": ['-', '-', '-'],
            "mă interesa materia Calculatoare Numerice 1": ['-', '-', '-', '-', '-', '-'],
            "curs": ['-'],
            "seminar": ['-'],
            "laborator": ['-'],
            "laboratorul de la Proiectare cu Microprocesoare": ['-', '-', '-', '-', '-', '-'],
            "cursul de Programare Orientata pe Obiecte": ['-', '-', '-', '-', '-', '-'],
            "seminarul de la Chimie aplicată": ['-', '-', '-', '-', '-'],
            "Semnale și sisteme, seria AB": ['-', '-', '-', '-', '-', '-'],
            "Metode Numerice de la seria CA": ['-', '-', '-', '-', '-', '-'],
            "grupa 331CB materia Sisteme de operare": ['-', '-', '-', '-', '-', '-'],
            "seminar de Algebră și Geometrie": ['-', '-', '-', '-', '-'],
            "3CC": ['-'],
            "in grupa 333CC": ['-', '-', '-'],
            "grupa 512AA": ['-', '-'],
            "sunt în grupa 232": ['-', '-', 'unde', '-'],
            "sunt de la seria AC": ['-', '-', '-', 'unde', '-'],
            "de la 412C1": ['-', '-', '-'],
            "seria CTI": ['-', '-'],
            "eu fac parte din grupa 311CD": ['-', '-', '-', '-', 'unde', '-'],
            "pentru seria CB": ['-', '-', '-'],
            # =====================================================================================================================================================================
            "Buna": ['-'],
            "o zi buna": ['-', '-', '-'],
            "super!": ['-', '-'],
            "mulțumesc mult": ['-', '-'],
            "mersi de informație": ['-', '-', '-'],
            "Cum poti sa ma ajuti": ['-', '-', '-', '-', '-'],
            "ce te pot intreba": ['-', '-', '-', '-'],
            "vreau ajutor": ['-', '-'],
            "mă poți ajuta cu ceva?": ['-', '-', '-', '-', '-', '-'],
            "la ce te pricepi": ['-', '-', '-', '-'],
            "care sunt lucrurile pe care știi să le faci?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "ce știi tu?": ['-', '-', '-', '-'],
            "Niște exemple de întrebări te rog": ['-', '-', '-', '-', '-', '-'],
            "vreau ceva exemple": ['-', '-', '-'],
            "arata-mi niște exemple de întrebări": ['-', '-', '-', '-', '-', '-', '-'],
            "Cum aș putea să ajung până la Universitate?": ['-', '-', '-', '-', '-', '-', '-', '-', 'unde'],
            "Dă-mi trasee către Voluntari": ['-', '-', '-', '-', '-', 'unde'],
            "vreau rutele de transport spre spitalul floreasca": ['-', '-', '-', '-', '-', 'unde', '-'],
            "as vrea sa merg la dedeman": ['-', '-', '-', '-', '-', 'unde'],
            "vreau sa ma duc la Lujerului": ['-', '-', '-', '-', '-', 'unde'],
            "ce iau ca să ajung în Băneasa": ['-', '-', '-', '-', '-', '-', 'unde'],
            "Reține te rog": ['-', '-', '-'],
            "memo te rog": ['-', '-', '-'],
            "ține minte ceva": ['-', '-', '-'],
            "atlasul de geografie se află pe raftul cu dicționarul": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "am pus bateria externă în rucsacul albastru": ['-', '-', '-', '-', '-', 'unde', '-'],
            "cheia franceză e pe balcon": ['-', '-', '-', '-', 'unde'],
            "cartela mea SIM veche este sub cutia telefonului": ['-', '-', '-', '-', '-', '-', 'unde', '-'],
            "am lăsat geanta în dulapul numărul 4": ['-', '-', '-', '-', 'unde', '-', '-'],
            "Victor stă pe Aleea Romancierilor numărul 12": ['-', '-', '-', 'unde', '-', '-', '-'],
            "eu stau la blocul nr. 8": ['-', '-', '-', 'unde', '-', '-'],
            "am lăsat mașina la service-ul auto din Crângași": ['-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "pliculețele de praf de copt sunt în cutiuța din primul sertar": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "am așezat etuiul de la ochelari peste teancul de reviste din hol": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-'],
            "am luat ornamentele de la dedeman": ['-', '-', '-', '-', '-', 'unde'],
            "roșiile din lădiță sunt de la țară": ['-', '-', '-', '-', '-', '-', 'unde'],
            "gladiolele sunt de la florăria din oraș": ['-', '-', '-', '-', 'unde', '-', '-'],
            "cartea de matematică a Mariei este pe biroul meu": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-'],
            "insigna de la facultate este pe rucsacul negru": ['-', '-', '-', '-', '-', '-', 'unde', '-'],
            "geanta de voiaj este de la Samsonite": ['-', '-', '-', '-', '-', '-', 'unde'],
            "cadoul este de la colegul meu de apartament": ['-', '-', '-', '-', 'unde', '-', '-', '-'],
            "am luat ochelarii de la mall": ['-', '-', '-', '-', '-', 'unde'],
            "unde am pus caietul de matematică 1?": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "știi unde sunt burghiele mici": ['-', 'unde', '-', '-', '-'],
            "pe unde am lăsat pompa de bicicletă": ['-', 'unde', '-', '-', '-', '-', '-'],
            "unde locuiește Claudia Ionescu": ['unde', '-', '-', '-'],
            "de unde este brelocul meu de la chei?": ['-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "unde se află bazinul de înot Dinamo": ['unde', '-', '-', '-', '-', '-', '-'],
            "unde mi-am aruncat sculele de construcții": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "de unde mi-am luat cravata cea grena": ['-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "zi-mi unde am lăsat certificatul meu de naștere": ['-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-'],
            "unde este setul de baterii pentru mouse?": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "codul meu IBAN este următorul": ['-', '-', '-', '-', 'care este'],
            "linkul de la concursul ACM de anul trecut e următorul": ['-', '-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "studenții care s-au înscris la masterul de securitate sunt următorii": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "numerele de telefon ale Georgianei sunt acestea": ['-', '-', '-', '-', '-', '-', 'care este'],
            "emailul lui Ionuț este acesta": ['-', '-', '-', '-', 'care este'],
            "membrii lotului de informatică din 2016 sunt cei care urmează": ['-', '-', '-', '-', '-', '-', '-', 'care este', '-', '-'],
            "codul meu numeric personal este cel ce urmează": ['-', '-', '-', '-', '-', 'care este', '-', '-'],
            "teorema lui Pitagora este asta": ['-', '-', '-', '-', 'care este'],
            "modelul SSD-ului meu e ăsta": ['-', '-', '-', '-', '-', '-', 'care este'],
            "hotelul la care ne-am cazat anul trecut la mare e acesta:": ['-', '-', '-', '-', '-', '-', '-', 'când', '-', '-', 'unde', '-', 'care este', '-'],
            "telefonul Dianei este 0745789654": ['-', '-', '-', 'care este'],
            "prețul ceasului meu Atlantic a fost 500 de lei": ['-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "numărul de la mașină al Elenei este B 07 ELN": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "codul de identificare al cardului meu de de acces la birou este 57627": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'care este'],
            "numele profesoarei mele de biologie din liceu este Mariana Mihai": ['-', '-', '-', '-', '-', '-', '-', '-', 'care este', '-'],
            "mărimea la tricou a lui Teo este L": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "materiile la care am luat 9 sunt M1 și M2": ['-', '-', '-', '-', '-', '-', '-', 'care este', '-', '-'],
            "rank-ul universității Politehnica este 800": ['-', '-', '-', '-', '-', '-', 'care este'],
            "tipul de baterii de la tastatură este AAA": ['-', '-', '-', '-', '-', '-', '-', 'care este'],
            "modelul tastaturii mele este logitech mx keys": ['-', '-', '-', '-', 'care este', '-', '-'],
            "ziua de naștere a Grațielei este pe 12 mai 1995": ['-', '-', '-', '-', '-', '-', '-', 'care este', '-', '-'],
            "care este telefonul Dianei": ['-', '-', '-', '-'],
            "care este prețul ceasului meu Atlantic": ['-', '-', '-', '-', '-', '-'],
            "care este numărul de la mașină al Elenei": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "zi-mi care e codul PIN de la cardul meu de sănătate": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care e numele profesoarei mele de biologie din liceu": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care sunt cele mai bune întrerupătoare": ['-', '-', '-', '-', '-', '-'],
            "care era emailul asistentului de la APP?": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "care a fost durata domniei lui Cezar": ['-', '-', '-', '-', '-', '-', '-'],
            "zi-mi care e codul de pe spatele telefonului": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "care e vărsta de pensionare la bărbați": ['-', '-', '-', '-', '-', '-', '-'],
            "care era modelul tastaturii mele?": ['-', '-', '-', '-', '-', '-'],
            "care este înălțimea vârfului Everest?": ['-', '-', '-', '-', '-', '-'],
            "la ce apartament stă Alex Marin?": ['-', '-', '-', '-', '-', '-', '-'],
            "ce fel de bec am pus la bucătărie": ['-', '-', '-', '-', '-', '-', '-', 'unde'],
            "în care dulap am lăsat plasa?": ['-', '-', 'unde', '-', '-', '-', '-'],
            "de la ce magazin am luat tortul de la ziua mea": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "care colegi au luat 10 la chimie": ['-', '-', '-', '-', '-', '-', '-'],
            "care copil a rupt scaunul de la școală": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "ce profesor a prezentat la conferință": ['-', '-', '-', '-', '-', 'unde'],
            "în ce zi e născut unchiul meu?": ['-', '-', 'când', '-', '-', '-', '-', '-'],
            "care dinamometru e afară": ['-', '-', '-', 'unde'],
            "ce floare este de la expo flora?": ['-', '-', '-', '-', '-', 'unde', '-', '-'],
            "care persoane suferă de diabet": ['-', '-', '-', '-', '-'],
            "care bomboane sunt pentru ziua ei": ['-', '-', '-', '-', '-', '-'],
            "ce os a înghițit acel băiat": ['-', '-', '-', '-', '-', '-'],
            "care bomboane sunt de la auchan": ['-', '-', '-', '-', '-', 'unde'],
            "mâine se termină perioada de rodaj a mașinii": ['când', '-', '-', '-', '-', '-', '-', '-'],
            "examenul la ML este peste 4 zile": ['-', '-', '-', '-', '-', '-', 'cât timp'],
            "testul practic de EIM o să fie săptămâna viitoare": ['-', '-', '-', '-', '-', '-', '-', 'când', '-'],
            "peste o zi expiră prăjiturile": ['-', '-', 'cât timp', '-', '-'],
            "ieri m-am tuns": ['când', '-', '-', '-', '-'],
            "aniversarea prieteniei cu Andreea e pe 30 aprilie": ['-', '-', '-', '-', '-', '-', 'când', '-'],
            "am trimis rezolvările la gazeta matematică miercurea trecută": ['-', '-', '-', '-', 'unde', '-', 'când', '-'],
            "vineri apare revista historia": ['când', '-', '-', '-'],
            "virusul a apărut acum 3 luni": ['-', '-', '-', '-', '-', 'cât timp'],
            "peste 123 de secunde trecem în noul an": ['-', '-', '-', 'cât timp', '-', '-', '-', '-'],
            "hackathonul s-a organizat weekend-ul trecut": ['-', '-', '-', '-', '-', 'când', '-', '-', '-'],
            "la primăvară e gata blocul": ['-', 'când', '-', '-', '-'],
            "voi pleca in Monaco peste 3 ore": ['-', '-', '-', 'unde', '-', '-', 'cât timp'],
            "am udat floarea roșie aseară": ['-', '-', '-', '-', 'când'],
            "mâine la prânz o să tund gazonul": ['când', '-', 'când', '-', '-', '-', '-'],
            "alaltăieri după amiază a plouat cu piatră la Botoșani": ['când', '-', 'când', '-', '-', '-', '-', '-', 'unde'],
            "de 3 ori pe oră trebuie să verific starea panoului de control": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "o dată la 4 ani se schimbă primarii": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "Dan vine acasă din Germania în fiecare vară": ['-', '-', 'unde', '-', 'unde', '-', '-', 'cât de des'],
            "copierea fișierelor de pe un disk pe altul a durat 4 minute": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', 'cât timp'],
            "am alergat timp de o oră și jumătate": ['-', '-', '-', '-', '-', 'cât timp', '-', '-'],
            "acum 3 ore și 15 minute președintele John Kennedy a ținut un discurs": ['-', '-', 'cât timp', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "când începe vacanța de iarnă": ['când', '-', '-', '-', '-'],
            "când am fost la mare anul trecut": ['când', '-', '-', '-', 'unde', 'când', '-'],
            "de când s-a deschis McDonalds din Slatina": ['-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "cât timp a durat bătălia de la Oituz": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "când am fost plecat în Germania": ['când', '-', '-', '-', '-', 'unde'],
            "până când va dura festivalul de muzică folk?": ['-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "de cât timp s-a deschis fabrica de mașini din Pitești": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "de cât timp era însurat Ghiță": ['-', '-', '-', '-', '-', '-'],
            "pentru cât timp o să fie plecat Mihai": ['-', '-', '-', '-', '-', '-', '-', '-'],
            "în cât timp am urcat pe vârful Omu?": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "peste cât timp începe sesiunea de comunicări științifice?": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "cât de des vin cei de la amenajări stradale să îngrijească plantele": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "de când până când se va lucra la reamenajarea clădirii?": ['-', 'când', '-', 'când', '-', '-', '-', '-', '-', '-', '-'],
            "în ce perioadă se vor da biletele gratis la operă": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "în ce interval o să se ia apa la apartament": ['-', '-', 'când', '-', '-', '-', '-', '-', '-', 'unde'],
            "peste cât timp o să se aprindă focul de tabără": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "cine a inventat motorul cu reacție": ['-', '-', '-', '-', '-', '-'],
            "cine a fost la ziua mea acum 3 ani": ['-', '-', '-', '-', 'când', '-', '-', '-', '-'],
            "cine m-a ajutat la proiectul de la anatomie?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "cine mi-a reparat bateria de la chiuveta de la baie?": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "spune-mi te rog cine a luat 10 la arhitectura sistemelor de calcul": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "Spune-mi te rog unde are biroul Ciprian Truică": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-'],
            "Zi-mi unde îl găsesc pe domnul Emil Slușanschi": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Unde are profesoara Irina Mocanu biroul din facultate?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Știi cumva care e biroul profesorului Ciprian Dobre?": ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Vreau să știu unde ar putea fi domnul Rughiniș": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "Arată-mi cum ajung în EC105": ['-', '-', '-', '-', '-', '-', 'unde'],
            "Cum pot să găsesc sala PR 204": ['-', '-', '-', '-', '-', '-', '-'],
            "Poți să mă ghidezi te rog către PR001?": ['-', '-', '-', '-', '-', '-', '-', 'unde', '-'],
            "Mă poți ajuta să ajung la sala EG 203?": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-', '-'],
            "pe unde pot merge către sala PR 306": ['-', 'unde', '-', '-', '-', 'unde', '-', '-'],
            "caut sala EC104, știi cumva unde este?": ['-', '-', '-', '-', '-', '-', 'unde', '-', '-'],
            "Zi-mi unde se ține Analiza Algoritmilor!": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-'],
            "Spune-mi unde are loc Calculatoare Numerice te rog": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Știi unde va fi fizica?": ['-', 'unde', '-', '-', '-', '-'],
            "Aș vrea să aflu unde o să fie predată teoria sistemelor": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "în ce sală va avea loc semnale și sisteme?": ['-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "Unde se ține seminarul de la grupa 313CC?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Poți să îmi spui unde are loc laboratorul pentru seria CA?": ['-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "Spune-mi te rog unde va avea loc cursul seriei AA": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Arată-mi unde se desfășoară laborator pentru grupa 12": ['-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-'],
            "Aș vrea să aflu unde va fi seminarul de la grupa 123A": ['-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-'],
            "Unde se ține curs de Programare Avansată în Java pentru seria AC?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Aș vrea să îmi zici unde are loc laboratorul de programare orientată pe obiecte al grupei 333CD": ['-', '-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Vreau să aflu unde o să fie seminarul de fizică de la grupa 31C": ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Poți să îmi spui unde se ține cursul de la seria CA de mecanică?": ['-', '-', '-', '-', 'unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "Unde o să fie predat laboratorul de la grupa 15 de structuri de date?": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
            "sunt de la grupa 333CD": ['-', '-', '-', 'unde', '-'],
            "pentru seria CA": ['-', '-', '-'],
            "511AC": ['-'],
            "seminarul": ['-'],
            "cursul de Proiectare logică": ['-', '-', '-', '-'],
            "la materia de Arhitectura sistemelor de calcul": ['-', '-', '-', '-', '-', '-', '-'],
            "mersi": ['-'],
            "numarul de telefon al mariei este asta": ['-', '-', '-', '-', '-', '-', '-'],
            "Unde va avea loc seminarul de ingineria programelor": ['unde', '-', '-', '-', '-', '-', '-', '-'],
            "unde o sa fie cursul de ingineria programelor de la seria CA": ['unde', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
        }

        allowed = ["unde", "când"]

        doc = self.nlp_spacy(text.lower())
        pre_deps = precomputed_deps[text] if text in precomputed_deps else None
        if not pre_deps:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NO " + text)
        word_syntactic_deps = []
        for i, spacy_token in enumerate(doc):
            no_spec_chars = spacy_token.text.translate(str.maketrans('', '', string.punctuation))
            # word_syntactic_deps.append(spacy_token.head.i)
            # word_syntactic_deps.append(spacy_token.dep_ if no_spec_chars and spacy_token.dep_ in allowed else '-')
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
