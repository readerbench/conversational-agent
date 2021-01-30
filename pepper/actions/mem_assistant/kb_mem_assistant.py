from actions.sparql_client import execute_sparql_query, execute_sparql_update, KB_BASE_URI
from actions.mem_assistant.types import MemAssistInfoType
from uuid import uuid4 as uuid
import logging

logger = logging.getLogger(__name__)

MEM_ASSIST_URI = f"{KB_BASE_URI}/mem_assistant#"
RDF_PREFIXES = f'PREFIX : <{MEM_ASSIST_URI}>'

insert_data = """
    INSERT DATA {{ {} }};
""".format

select = """
    SELECT {} {{ {} }}
""".format


class QueryBuilder:

    @staticmethod
    def query_create_noun_phrase(entity, match_existing=False):
        """ Build a SPARQL query that inserts a new entity into the database. """

        eid = uuid().hex
        string = (entity.get('pre', "") + " " + entity['value']).strip()

        if not entity['specifiers']:
            # single node noun phrase
            query = f"""
                :{eid} 
                    a :Class ;
                    :value :{entity["lemma"]} ;
                    :pre "{entity.get("pre", "")}" .
            """  # TODO merge or create
        else:
            # instance node along with some specifiers
            cls_id = uuid().hex

            query = f"""
                :{cls_id} 
                    a :Class ;
                    :value :{entity["lemma"]} ;
                    :pre "{entity.get("pre", "")}" .
                :{eid} 
                    a :Instance ;
                    :IS_A :{cls_id} .
            """  # TODO merge or create

            for spec in entity['specifiers']:
                inner_query, inner_id, inner_str = QueryBuilder.query_create_noun_phrase(spec, True)
                query += inner_query

                # link the nodes
                if spec['question'] in ['care', 'ce fel de']:
                    query += f"""
                        :{eid} :SPEC :{inner_id} . 
                    """
                elif spec['question'] == 'al cui':
                    query += f"""
                        :{inner_id} :HAS :{eid} . 
                    """

                string += " " + inner_str

            query += f""" 
                :{eid} :value "{string}" . 
            """

        return query, eid, string

    @staticmethod
    def query_match_noun_phrase(entity):
        """ Build a SPARQL query that tries to match (find) an entity in the database. """

        eid = uuid().hex

        # match the entity as a simple node or as an instance node
        query = f"""
            ?{eid} :IS_A? [
                :value :{entity["lemma"]} ;
                :pre "{entity.get("pre", "")}"
            ] .
        """

        for spec in entity['specifiers']:
            inner_query, inner_id = QueryBuilder.query_match_noun_phrase(spec)
            query += inner_query

            # link the specifier nodes
            if spec['question'] in ['care', 'ce fel de']:
                query += f"""
                    ?{eid} :SPEC ?{inner_id} .
                """
            elif spec['question'] == 'al cui':
                query += f"""
                    ?{inner_id}) :HAS ?{eid} .
                """

        return query, eid


def value(entry):
    val = entry['value']
    if entry['type'] == "uri":
        val = val.split("#")[1]
    return val


class DbBridge:
    def __init__(self):
        pass

    def set_value(self, entity, value, type=MemAssistInfoType.VAL):
        """ Store a detail of an entity in the knowledge base. """

        query, node_id, _ = QueryBuilder.query_create_noun_phrase(entity, match_existing=True)

        if type == MemAssistInfoType.VAL:
            query += f""" 
                :{node_id} :{type.value} [:value "{value}"] .
            """
        elif type == MemAssistInfoType.LOC:
            query_create_location, location_node_id, _ = QueryBuilder.query_create_noun_phrase(value)
            query += query_create_location
            query += f""" 
                :{node_id} :{type.value} :{location_node_id} .
            """
        elif type in [MemAssistInfoType.TIME_POINT,
                      MemAssistInfoType.TIME_START,
                      MemAssistInfoType.TIME_END,
                      MemAssistInfoType.TIME_RANGE,
                      MemAssistInfoType.TIME_DURATION]:
            query += f""" 
                :{node_id} :{type.value} [
                    a :time ;
                    :value "{value}"
                ] .
            """

        query = RDF_PREFIXES + insert_data(query)
        logger.debug(query)
        execute_sparql_update(query)

    def get_value(self, entity, type=MemAssistInfoType.VAL):
        """ Get a detail of an entity from the knowledge base. """

        query, node_id = QueryBuilder.query_match_noun_phrase(entity)
        query += f""" 
            ?{node_id}
                :value ?entity ;
                :{type.value} [:value ?val] .
        """

        query = RDF_PREFIXES + select(f"?entity ?val", query)
        logger.debug(query)
        result = execute_sparql_query(query)
        values = [[value(record['val']), value(record['entity'])] for record in result]

        return self.__prettify_result(values)

    def store_action(self, components):
        """
        Store a complete action of a subject, eventually together with other semantic entities
        (location, timestamp, direct object, etc.).
        """

        query, subj_node_id, _ = QueryBuilder.query_create_noun_phrase(components['subj'], match_existing=True)

        act = uuid().hex
        query += f"""
            :{subj_node_id} :ACTION [
                a :action ;
                :value: "{components["action"]}"
            ] .
         """

        if components['ce']:
            sub_query, node_id, _ = QueryBuilder.query_create_noun_phrase(components['ce'])
            query += sub_query
            query += f""" 
                :{act} :CE :{node_id} .
            """

        if components['loc']:
            for loc in components['loc']:
                query_create_location, location_node_id, _ = QueryBuilder.query_create_noun_phrase(loc)
                query += query_create_location
                query += f""" 
                    :{act} :LOC :{location_node_id} .
                """

        if components['time']:
            for i, time in enumerate(components['time']):
                query += f""" 
                    :{act} :{time[1].value} [
                        a :time ;
                        :value: "{time[0]}"
                    ] .
                """

        query = RDF_PREFIXES + insert_data(query)
        logger.debug(query)
        result = execute_sparql_query(query)

    def get_action_time(self, components, info_type=MemAssistInfoType.TIME_POINT):
        """
        Get timestamp of an action expressed through a complex sentence
        (containing more than the entity whose time is requested).
        """

        # try to find the subject of the action
        query, subj_node_id = QueryBuilder.query_match_noun_phrase(components['subj'])

        # match the requested action
        query += f"""
            ?{subj_node_id} :ACTION ?act .
            ?act :value "{components["action"]}" . 
        """

        noun_phrase_nodes = [subj_node_id]
        if components['ce']:
            sub_query, node_id = QueryBuilder.query_match_noun_phrase(components['ce'])
            query += sub_query
            query += f"""
                ?act :CE ?{node_id} .
            """
            noun_phrase_nodes.append(node_id)

        if components['loc']:
            for loc in components['loc']:
                query_create_location, location_node_id = QueryBuilder.query_match_noun_phrase(loc)
                query += query_create_location
                query += f"""
                    ?act :LOC ?{location_node_id} .
                """
                noun_phrase_nodes.append(location_node_id)

        if components['time']:
            for i, time in enumerate(components['time']):
                query += f"""
                    ?act :{time[1].value}  [ :value: "{time[0]}" ] .
                """

        # extract the requested property of the action
        query += f"""
            ?act :{info_type.value} ?time .
        """

        query = RDF_PREFIXES + select('*', query)
        logger.debug(query)
        result = execute_sparql_query(query)
        values = [[value(record['time'])] + [value(record[np]) for np in noun_phrase_nodes] for record in result]

        return self.__prettify_result(values)

    @staticmethod
    def __prettify_result(values):
        if not values:
            return "Nu știu"
        if len(values) == 1:
            return values[0][0]
        return '\n'.join([f"▪ {', '.join(val[1:])}: ➜ {val[0]}" for val in values])
