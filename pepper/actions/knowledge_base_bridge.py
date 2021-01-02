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


def get_professor_offices():
    query = """
        PREFIX : <%s>
        select ?name ?office {
            ?prof a :Professor .
            ?prof :name ?name .
            ?prof :office ?office .
        }
    """ % graphdb_base_uri

    result = execute_sparql_query(query)

    professors = [{
        'name': professor['name']['value'],
        'office': professor['office']['value']
    } for professor in result]

    return professors


def get_activities_schedule():
    query = """
        PREFIX : <%s>
        select ?id ?name ?type ?room ?teacher ?group ?semigroup {
            ?activity a :Activity ;
                        :id ?id ;
                        :name ?name ;
                        :type ?type ;
                        :room ?room .
            optional { ?activity :teacher ?teacher } .
            ?activity :groups [
                :group ?group;
                :semigroup ?semigroup 
            ] .
            ?activity :timeSlot [
                :day ?day;
                :time ?time;
                :duration ?duration;
            ] .
        }
    """ % graphdb_base_uri

    result = execute_sparql_query(query)

    activities = []
    for activity in result:
        activity_parsed = {k: v['value'] for (k, v) in activity.items()}
        if activity_parsed['room']:
            activity_parsed['room'] = activity_parsed['room'].split('#')[1]
        activities.append(activity_parsed)

    return activities


def get_rooms():
    query = """
        PREFIX : <%s>
        select ?id ?direction ?room { 
        ?room a :Room ;
                :id ?id ;
                :direction ?direction .
    } 
    """ % graphdb_base_uri

    result = execute_sparql_query(query)

    rooms = [{k: v['value'] for (k, v) in activity.items()} for activity in result]

    return rooms
