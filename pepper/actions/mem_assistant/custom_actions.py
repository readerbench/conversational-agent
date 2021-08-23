"""
Custom actions to be performed in response to specific intents.
"""

from typing import Any, Text, Dict, List, Union
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.forms import FormAction
import re

from actions.mem_assistant.kb_mem_assistant import DbBridge
from actions.mem_assistant.types import MemAssistInfoType

db_bridge = DbBridge()
entity_extraction_failure_msg = "Nu am putut extrage entitățile"

SEMANTIC_ROLES = "semantic_roles"


class ActionStoreAttribute(Action):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_store_attr"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # get user's utterance (request)
        message = tracker.latest_message

        # extract relevant entities from the phrase
        entity = None
        value = None

        semantic_roles = message.get(SEMANTIC_ROLES)
        print(semantic_roles)
        for ent in semantic_roles:
            if ent['question'] == 'cine':
                entity = ent
            elif ent['question'] == 'care este':
                value = ent['value']

        # insert data into the database
        if entity and value:
            success = db_bridge.set_value(entity, value)
            if not success:
                dispatcher.utter_message(template="utter_error_storing_data")
        else:
            dispatcher.utter_message(entity_extraction_failure_msg)

        return []


class ActionGetAttribute(Action):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_get_attr"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # get user's utterance (request)
        message = tracker.latest_message

        # extract relevant entities from the phrase
        entity = None

        semantic_roles = message.get(SEMANTIC_ROLES)
        print(semantic_roles)
        for ent in semantic_roles:
            if ent['question'] in ['ce', 'cine'] and ent['value'] != "care":
                entity = ent

        # query the database
        if entity:
            result = db_bridge.get_value(entity)
            if result:
                dispatcher.utter_message(result)
            else:
                dispatcher.utter_message(template="utter_error_getting_data")
        else:
            dispatcher.utter_message(entity_extraction_failure_msg)
        return []


class ActionStoreLocation(Action):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_store_location"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = tracker.latest_message

        # extract relevant entities from the phrase
        entity = None
        location = None

        semantic_roles = message.get(SEMANTIC_ROLES)
        print(semantic_roles)
        for ent in semantic_roles:
            if ent['question'] == 'ce' or (ent['question'] == 'cine' and not entity):
                entity = ent
            elif ent['question'] == 'unde':
                location = ent

        # query the database
        if entity and location:
            success = db_bridge.set_value(entity, location, type=MemAssistInfoType.LOC)
            if not success:
                dispatcher.utter_message(template="utter_error_storing_data")
        else:
            dispatcher.utter_message(entity_extraction_failure_msg)
        return []


class ActionGetLocation(Action):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_get_location"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = tracker.latest_message

        # extract relevant entities from the phrase
        entity = None

        semantic_roles = message.get(SEMANTIC_ROLES)
        print(semantic_roles)
        for ent in semantic_roles:
            if ent['question'] == 'ce' or (ent['question'] == 'cine' and not entity):
                entity = ent

        if entity:
            result = db_bridge.get_value(entity, type=MemAssistInfoType.LOC)
            if result:
                dispatcher.utter_message(result)
            else:
                dispatcher.utter_message(template="utter_error_getting_data")
        else:
            dispatcher.utter_message(entity_extraction_failure_msg)
        return []


def get_time_type(ent, action):
    """ Determine the type of the timestamp: a specific point in time, a start point, an end point or a duration. """

    question = ent['question']
    phrase = (ent.get('pre', "") + " " + ent['ext_value']).strip()

    info_type = MemAssistInfoType.TIME_POINT

    if phrase.startswith("de ") or action in ["începe", "porni"]:
        info_type = MemAssistInfoType.TIME_START
    elif phrase.startswith("până ") or action in ["termina", "sfârși", "încheia", "finaliza"]:
        info_type = MemAssistInfoType.TIME_END
    elif question == "cât timp" and re.match(r'^(?:acum|peste|după|la) ', phrase):
        info_type = MemAssistInfoType.TIME_POINT
    elif question == "cât timp" or phrase in ["cât timp"]:
        info_type = MemAssistInfoType.TIME_DURATION

    return info_type


def extract_sentence_components(semantic_roles):
    """ Extract the main proposition parts of a sentence. """

    components = {
        'subj': '?',
        'action': '?',
        'ce': None,
        'loc': [],
        'time': []
    }

    action = None
    for ent in semantic_roles:
        if ent['question'] == "ROOT":
            action = ent['ext_value']

    for ent in semantic_roles:
        if ent['question'] == 'cine':
            components['subj'] = ent
        elif ent['question'] == 'ROOT':
            components['action'] = ent['ext_value']
        elif ent['question'] == 'ce':
            components['ce'] = ent
        elif ent['question'] == 'unde':
            components['loc'].append(ent)
        elif ent['question'] in ['când', 'cât timp']:
            if any(ask_particle in ent['ext_value'].split() for ask_particle in ['când', 'cât', 'ce', 'care']):
                # this entity is only used to formulate the question
                continue
            info_type = get_time_type(ent, action)
            components['time'].append(((ent.get('pre', "") + " " + ent['ext_value']).strip(), info_type))

    return components


class ActionGetTime(Action):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_get_time"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = tracker.latest_message

        # extract relevant entities from the phrase
        entity = None
        action = None
        times = []
        is_simple_event = True  # simple event (noun phrase) or a complex one (containing subject/other details)

        semantic_roles = message.get(SEMANTIC_ROLES)

        for ent in semantic_roles:
            if ent['question'] in ['ce', 'cine']:
                if entity:
                    is_simple_event = False
                entity = ent
            elif ent['question'] in ['când', 'cât timp']:
                times.append(ent)
            elif ent['question'] == 'ROOT':
                action = ent['lemma']
            else:
                is_simple_event = False

        # determine the type of timestamp requested
        info_type = get_time_type(times[0], action)

        if entity:
            if is_simple_event:
                result = db_bridge.get_value(entity, type=info_type)
            else:
                sentence_components = extract_sentence_components(semantic_roles)
                result = db_bridge.get_action_time(sentence_components, info_type)
            if result:
                dispatcher.utter_message(result)
            else:
                dispatcher.utter_message(template="utter_error_getting_data")
        else:
            dispatcher.utter_message(entity_extraction_failure_msg)
        return []


class ActionStoreTime(Action):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_store_time"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = tracker.latest_message

        # extract relevant entities from the phrase
        entity = None
        action = None
        time = None
        is_simple_event = True

        semantic_roles = message.get(SEMANTIC_ROLES)

        for ent in semantic_roles:
            if ent['question'] in ['ce', 'cine']:
                if entity:
                    is_simple_event = False
                entity = ent
            elif ent['question'] in ['când', 'cât timp']:
                time = ent
            elif ent['question'] == "ROOT":
                action = ent['lemma']
            else:
                is_simple_event = False

        info_type = get_time_type(time, action)

        if entity:
            if is_simple_event:
                success = db_bridge.set_value(entity, (time['pre'] + " " + time['ext_value']).strip(), type=info_type)
            else:
                sentence_components = extract_sentence_components(semantic_roles)
                success = db_bridge.store_action(sentence_components)
            if not success:
                dispatcher.utter_message(template="utter_error_storing_data")
        else:
            dispatcher.utter_message(entity_extraction_failure_msg)
        return []


class ActionKeepRawAttrEntity(Action):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_keep_raw_attr_entity"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = tracker.latest_message

        entity = None
        semantic_roles = message.get(SEMANTIC_ROLES)
        for ent in semantic_roles:
            if ent['question'] == 'cine':
                entity = ent

        return [SlotSet("raw_attr_entity", entity)]


class RawDataStoreForm(FormAction):
    """Example of a custom form action"""

    def name(self):
        """Unique identifier of the form"""
        return "raw_data_store_form"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["raw_attr_val"]

    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

        return {
            "raw_attr_val": self.from_text(),  # the raw value is actually the whole text entered by the user
        }

    def submit(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[Dict]:
        """Define what the form has to do after all required slots are filled"""

        raw_attr_entity = tracker.get_slot("raw_attr_entity")
        raw_attr_val = tracker.get_slot("raw_attr_val")
        db_bridge.set_value(raw_attr_entity, raw_attr_val, type=MemAssistInfoType.VAL)
        return [SlotSet("raw_attr_val", None)]
