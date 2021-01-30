import requests
from os import environ
from enum import Enum
import logging

logger = logging.getLogger(__name__)

KB_API_URL = environ.get('KB_API_URL')
KB_UPDATE_API_URL = f"{KB_API_URL}/statements"
KB_BASE_URI = environ.get('KB_BASE_URI')


class QueryType(Enum):
    SELECT = 'query'
    UPDATE = 'update'


def execute_sparql_query(query):
    """ Perform a SPARQL query on the knowledge base. """

    headers = {
        'Accept': 'application/sparql-results+json'
    }
    query_params = {
        'name': 'Pepper query',
        'infer': True,
        'sameAs': True,
        'query': query
    }

    r = requests.get(KB_API_URL, params=query_params, headers=headers)
    # logger.debug("RESPONSE", r.text)
    return r.json().get('results', {}).get('bindings', None)


def execute_sparql_update(query):
    """ Perform a SPARQL update on the knowledge base. """

    headers = {
        'Accept': '*/*'
    }
    query_params = {
        'name': 'Pepper query',
        'infer': True,
        'sameAs': True,
        'update': query
    }

    r = requests.post(KB_UPDATE_API_URL, params=query_params, headers=headers)
    # logger.debug("RESPONSE", r.text)


def prefix_list_to_string(prefixes):
    result = ''
    for (prefix, uri) in prefixes:
        result += f'PREFIX {prefix} <{uri}>\n'
    return result
