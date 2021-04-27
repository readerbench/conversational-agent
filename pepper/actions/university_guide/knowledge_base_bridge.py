from actions.sparql_client import execute_sparql_query, KB_BASE_URI


def get_professor_offices():
    query = """
        PREFIX : <%s#>
        select ?name ?office {
            ?prof a :Professor .
            ?prof :name ?name .
            ?prof :office ?office .
        }
    """ % KB_BASE_URI

    result = execute_sparql_query(query)

    professors = [{
        'name': professor['name']['value'],
        'office': professor['office']['value']
    } for professor in result]

    return professors


def get_activities_schedule():
    query = """
        PREFIX : <%s#>
        
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
    """ % KB_BASE_URI

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
        PREFIX : <%s#>
        select ?id ?direction ?room { 
        ?room a :Room ;
                :id ?id ;
                :direction ?direction .
    } 
    """ % KB_BASE_URI

    result = execute_sparql_query(query)

    rooms = [{k: v['value'] for (k, v) in activity.items()} for activity in result]

    return rooms
