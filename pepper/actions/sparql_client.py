import requests
from os import environ
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

KB_API_URL = environ.get('KB_API_URL')
KB_UPDATE_API_URL = f"{KB_API_URL}/statements"
KB_BASE_URI = environ.get('KB_BASE_URI')


class QueryType(Enum):
    SELECT = 'query'
    UPDATE = 'update'


def get(query_params, headers, num_retries=5):
    try:
        logger.info(f"Sending DB request to {KB_API_URL}. Num retries left: {num_retries}")
        logger.info(f"Query: {query_params['query']}")

        res = requests.get(KB_API_URL, params=query_params, headers=headers)
        logger.info(f"DB response code {res.status_code}")

        if res.status_code < 400:
            return res.json().get('results', {}).get('bindings', None)
    except Exception as err:
        logger.error(f"DB query error: {err}")
        if num_retries > 0:
            time.sleep(5)
            return get(query_params, headers, num_retries - 1)
    return None


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

    return get(query_params, headers)


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

    try:
        logger.info(f"Sending DB POST request. Query: {query}")

        res = requests.post(KB_UPDATE_API_URL, params=query_params, headers=headers)
        logger.debug(f"DB response code {res.status_code}")
    except Exception as err:
        logger.error(f"DB update error: {err}")


def prefix_list_to_string(prefixes):
    result = ""
    for (prefix, uri) in prefixes:
        result += f"PREFIX {prefix} <{uri}>\n"
    return result
