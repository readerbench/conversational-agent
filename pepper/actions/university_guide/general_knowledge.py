import logging
import random
import requests
from typing import Any, Text, Dict, List

# from googletrans import Translator

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.utils import get_entities


logger = logging.getLogger(__name__)


class ActionGeneralKnowledge(Action):

    def name(self) -> Text:
        return "action_general_knowledge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entity = get_entities(tracker.latest_message, 'entity')
        if not entity:
            dispatcher.utter_message(template="utter_unknown")
            return []

        req = requests.get(url="https://api.duckduckgo.com", params={"q": entity['value'], "format": "json", "pretty": "1"}) 
        json = req.json() 

        logger.debug('json = ' + str(json))

        output = json["Abstract"]
        if not output and len(json["RelatedTopics"]):
            output = json["RelatedTopics"][0]["Text"]

        if not output:
            dispatcher.utter_message(template="utter_unknown")
            return []

        # translator = Translator().translate(output, src='en', dest='ro')
        # logger.debug('translator = ' + str(translator))

        dispatcher.utter_message(text=translator.text)
        return []
