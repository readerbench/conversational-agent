import requests
from os import environ

graphdb_api_url = environ.get('GRAPHDB_API_URL')
graphdb_base_uri = environ.get('GRAPHDB_BASE_URI')


def execute_sparql_query(query):
    """ Perform a SPARQL query on the GraphDB knowledge base. """

    headers = {
        'Accept': 'application/sparql-results+json'
    }
    query_params = {
        'name': 'SPARQL Select template',
        'infer': True,
        'sameAs': True,
        'query': query
    }
    r = requests.get(graphdb_api_url, params=query_params, headers=headers)
    return r.json().get('results', {}).get('bindings', None)


def prefix_list(prefixes):
    result = ''
    for (prefix, uri) in prefixes:
        result += f'PREFIX {prefix} <{uri}>\n'
    return result
