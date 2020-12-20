import logging
from unidecode import unidecode
import numpy as np
from itertools import permutations
from math import inf

logger = logging.getLogger(__name__)


def get_entities(message, entity_name):
    if not message['entities']:
        return None

    for entity in message['entities']:
        if entity['entity'] == entity_name:
            return entity

    return None


def get_entity_or_slot(tracker, name):
    entity = get_entities(tracker.latest_message, name)
    if entity:
        return entity['value']
    else:
        return tracker.get_slot(name)


def concept_to_dict(remote_concept):
    entity = {"_id": remote_concept.id, "_type": remote_concept.type().label()}

    for each in remote_concept.attributes():
        label = each.type().label()
        if label in entity:
            if not isinstance(entity[label], list):
                entity[label] = [entity[label]]
            entity[label].append(each.value())
        else:
            entity[label] = each.value()

    if remote_concept.is_relation():
        for role, players in remote_concept.role_players_map().items():
            # TODO: Should do as above because not all "players" are arrays
            entity[role.label()] = []
            for player in players:
                entity[role.label()].append(concept_to_dict(player))

    return entity


def get_grakn_entities(entity_type):
    query = f"match $x isa {entity_type}; get;"
    return []
    # with GraknClient(uri=URI) as client:
    #     with client.session(keyspace=KEYSPACE) as session:
    #         with session.transaction().read() as transaction:
    #             answer_iterator = transaction.query(query)
    #             concepts = [ans.get("x") for ans in answer_iterator]
    #             return [concept_to_dict(concept.as_remote(transaction)) for concept in concepts]


def editdistance(seq1, seq2):
    """ Compute levenshtein distance of two given strings. """

    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def match_multitoken_string(target, candidates, edit_dist_threshold=5):
    """
    Find matches of a (space-separated) string into a list of possible candidates.

    :param edit_dist_threshold Maximum allowed edit distance so that two strings to be considered the same
    :return The list of strings that match the target
    """

    target_tokens = unidecode(target.lower()).split(' ')

    distances = []
    for idx, candidate in enumerate(candidates):
        min_dist = inf

        candidate_tokens = unidecode(candidate.lower()).split(' ')
        if len(candidate_tokens) < len(target_tokens):
            # Supplement with empty tokens if the target string has more tokens
            candidate_tokens += ['' for _ in range(len(target_tokens) - len(candidate_tokens))]
        candidate_token_perms = permutations(candidate_tokens)

        # Find minimum edit-distance match between the tokens of the target and the candidate string, respectively
        for perm in candidate_token_perms:
            dist = sum([editdistance(target_tokens[i], perm[i]) for i in range(len(target_tokens))])
            min_dist = min(min_dist, dist)

        distances.append((idx, int(min_dist)))

    distances.sort(key=lambda d: d[1])
    logging.info(f'match_multitoken_string for "{target}" found as best matches: '
                 f'{[(candidates[dist[0]], dist[1]) for dist in distances[:5]]}')

    if distances[0][1] <= 2:
        # Perfect match
        return [idx for (idx, d) in distances if d == distances[0][1]]
    else:
        # Select only the "best" matches
        return [idx for (idx, d) in distances if d <= edit_dist_threshold and d - distances[0][1] <= 2]
