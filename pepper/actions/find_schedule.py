from datetime import datetime, timedelta
from difflib import SequenceMatcher
import logging
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.utils import get_entity_or_slot
from actions.knowledge_base_bridge import get_activities_schedule

logger = logging.getLogger(__name__)


def find_event(knowledge_base, course, group_name, class_type):
    best_event_ratio = 0.0
    best_event = None

    for e in knowledge_base:
        a = max(SequenceMatcher(None, e['name'].lower(), course.lower()).ratio(),
                SequenceMatcher(None, e['id'].lower(), course.lower()).ratio())
        b = SequenceMatcher(None, e['group'].lower(), group_name.lower()).ratio()
        c = SequenceMatcher(None, e['type'].lower(), class_type.lower()).ratio()
        event_ratio = a * b * c
        if best_event_ratio < event_ratio:
            best_event_ratio = event_ratio
            best_event = e

    if best_event_ratio > 0.5:
        return best_event

    return None


def parse_natural_delta(recurrence):
    value, unit = recurrence.split(' ')
    value = int(value)

    if unit.startswith('d'):
        return timedelta(days=value)
    elif unit.startswith('s'):
        return timedelta(seconds=value)
    elif unit.startswith('micro'):
        return timedelta(microseconds=value)
    elif unit.startswith('milli'):
        return timedelta(milliseconds=value)
    elif unit.startswith('min'):
        return timedelta(minutes=value)
    elif unit.startswith('h'):
        return timedelta(hours=value)
    elif unit.startswith('w'):
        return timedelta(weeks=value)
    else:
        raise 'Could not parse ' + recurrence


def to_natural_time(time):
    delta = time - datetime.now()
    seconds = delta.total_seconds()
    if seconds < 1 * 12 * 60 * 60:
        return "azi, la ora {}".format(time.strftime("%H:%M"))
    if seconds < 1 * 24 * 60 * 60:
        return "maine, la ora {}".format(time.strftime("%H:%M"))
    elif seconds < 7 * 24 * 60 * 60:
        days = ["Luni", "Marti", "Miercuri", "Joi", "Vineri", "Sambata", "Duminica"]
        return "{}, la ora {}".format(days[time.weekday()], time.strftime("%H:%M"))
    else:
        return str(time)


class ActionFindSchedule(Action):

    def __init__(self):
        super().__init__()
        self.classes = get_activities_schedule()

    def name(self) -> Text:
        return "action_find_schedule"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Extract the required slots from the dialog tracker
        course = get_entity_or_slot(tracker, 'course')
        group_name = get_entity_or_slot(tracker, 'group_name')
        class_type = get_entity_or_slot(tracker, 'class_type')

        logger.debug(f'Extracted course: {course} group: {group_name} class_type: {class_type}')
        event = None
        if course and group_name and class_type:
            event = find_event(self.classes, course, group_name, class_type)
        logger.debug(f"event = {str(event)}")

        if not event:
            events = []

            if course:
                events.append(SlotSet('course', course))
            if group_name:
                events.append(SlotSet('group_name', group_name))
            if class_type:
                events.append(SlotSet('class_type', class_type))

            if not course:
                dispatcher.utter_message(template="utter_ask_course")
                return events

            if not group_name:
                dispatcher.utter_message(template="utter_ask_group_name")
                return events

            if not class_type:
                dispatcher.utter_message(template="utter_ask_class_type")
                return events

            dispatcher.utter_message(template="utter_no_info", entity="activitatea")
            return events

        # TODO
        event_time = event['start_date']
        now_time = datetime.now()
        delta = parse_natural_delta(event['recurrence'])

        while event_time < now_time:
            event_time = event_time + delta

        dispatcher.utter_message(
            template="utter_schedule",
            course=course,
            group_name=group_name,
            class_type=class_type,
            room=event['room'],
            time=to_natural_time(event_time),
        )
        return []
