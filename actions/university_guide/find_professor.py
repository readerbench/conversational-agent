import logging
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.utils import get_entities, match_multitoken_string
from actions.university_guide.knowledge_base_bridge import get_professor_offices

logger = logging.getLogger(__name__)


class ActionFindProfessor(Action):

    def __init__(self):
        super().__init__()
        self.professors = get_professor_offices()

    def name(self) -> Text:
        return "action_find_professor"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extract the <professor> entity from the user utterance
        professor_entity = get_entities(tracker.latest_message, 'professor')
        logger.debug('<professor> entity: ' + str(professor_entity))

        if not professor_entity:
            dispatcher.utter_message(template="utter_entity_not_detected", entity="numele unui profesor")
            return []

        matches = match_multitoken_string(professor_entity['value'], [prof['name'] for prof in self.professors])

        if not matches:
            dispatcher.utter_message(template="utter_no_info", entity="profesorul")
        elif len(matches) == 1:
            # The professor was uniquely identified
            dispatcher.utter_message(
                template="utter_professor_office",
                professor=self.professors[matches[0]]['name'],
                office=self.professors[matches[0]]['office']
            )
        else:
            # Multiple close matches. Ask the user to choose between them
            # TODO wait for user choice
            dispatcher.utter_message(
                template="utter_select_from_multiple_matches",
                options=', '.join([self.professors[i]['name'] for i in matches])
            )

        return []
