from difflib import get_close_matches
import logging
import random
from typing import Any, Text, Dict, List
import numpy as np
import re

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.utils import get_entities
from .knowledge_base_bridge import get_rooms

logger = logging.getLogger(__name__)


def find_room(rooms, room_name):
    # Format the room name to follow the naming format from the knowledge base
    room_name = re.sub(r'[^\w0-9]', '', room_name).upper()

    names = get_close_matches(room_name, map(lambda x: x['id'], rooms), n=1)
    if len(names) == 0:
        return None

    for room in rooms:
        if room['id'] == names[0]:
            return room

    return None


class ActionFindRoom(Action):

    def __init__(self):
        super().__init__()
        self.rooms = get_rooms()

    def name(self) -> Text:
        return "action_find_room"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        room_entity = get_entities(tracker.latest_message, 'room')
        if not room_entity:
            dispatcher.utter_message(template="utter_entity_not_detected", entity="numele sălii")
            return []

        room = find_room(self.rooms, room_entity['value'])
        if not room:
            dispatcher.utter_message(template="utter_no_info", entity="sala")
            return []

        floor = re.match('[^0-9]*([0-9])', room['id'])
        floor = int(floor[1]) if floor else '?'
        directions = [f"la {'parter' if floor == 0 else 'etajul ' + str(floor)}"]
        directions_str = [r['direction'] for r in self.rooms if r['id'] == room['id']]

        # Select a few more directions
        for _ in range(1 + int(random.random() / 0.25)):
            if len(directions_str) == 0:
                break
            idx = np.random.choice(len(directions_str))
            directions.append(directions_str[idx])
            del directions_str[idx]

        if len(directions) > 1:
            directions = "{} si {}".format(', '.join(directions[:-1]), directions[-1])
        else:
            directions = directions[0]

        dispatcher.utter_message(
            template="utter_room_location",
            room_input=room['id'],
            direction=directions
        )

        return []
