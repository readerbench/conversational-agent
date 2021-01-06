from graphdb_client import execute_sparql_query, graphdb_base_uri
from actions.types import MemAssistInfoType


class QueryBuilder:
    id = 0

    @staticmethod
    def reset():
        QueryBuilder.id = 0

    @staticmethod
    def query_create_noun_phrase(entity, match_existing=False):
        """ Build a GraphDB query that inserts a new entity into the database. """

        action = "merge" if match_existing else "create"

        QueryBuilder.id += 1
        eid = f'n{QueryBuilder.id}'
        string = (entity.get('pre', "") + " " + entity['value']).strip()

        if not entity['specifiers']:
            # single node noun phrase
            query = f' {action} ({eid}:class {{value: "{entity["lemma"]}", pre: "{entity.get("pre", "")}"}})'
        else:
            # instance node along with some specifiers
            QueryBuilder.id += 1
            cls_id = f'n{QueryBuilder.id}'

            query = f' {action} ({cls_id}:class {{value: "{entity["lemma"]}", pre: "{entity.get("pre", "")}"}})' \
                    f' create ({eid}:instance)-[:IS_A]->({cls_id})'

            for spec in entity['specifiers']:
                inner_query, inner_id, inner_str = \
                    QueryBuilder.query_create_noun_phrase(spec, True)
                query += inner_query

                # link the nodes
                if spec['question'] in ['care', 'ce fel de']:
                    query += f' create ({eid})-[:SPEC]->({inner_id})'
                elif spec['question'] == 'al cui':
                    query += f' create ({inner_id})-[:HAS]->({eid})'

                string += " " + inner_str

            query += f' set {eid}.value = "{string}"'

        return query, eid, string

    @staticmethod
    def query_match_noun_phrase(entity):
        """ Build a Neo4j query that tries to match (find) an entity in the database. """

        QueryBuilder.id += 1
        eid = f'n{QueryBuilder.id}'

        # match the entity as a simple node or as an instance node
        query = f' match ({eid})-[:IS_A*0..1]->({{value: "{entity["lemma"]}", pre: "{entity.get("pre", "")}"}})'

        for spec in entity['specifiers']:
            inner_query, inner_id = QueryBuilder.query_match_noun_phrase(spec)
            query += inner_query

            # link the specifier nodes
            if spec['question'] in ['care', 'ce fel de']:
                query += f' match ({eid})-[:SPEC]->({inner_id})'
            elif spec['question'] == 'al cui':
                query += f' match ({inner_id})-[:HAS]->({eid})'

        return query, eid


class DbBridge:
    def __init__(self):
        pass

    def set_value(self, entity, value, type=MemAssistInfoType.VAL):
        """ Store a detail of an entity in the database. """

        query, node_id, _ = QueryBuilder.query_create_noun_phrase(entity, match_existing=True)

        if type == MemAssistInfoType.VAL:
            query += f' create ({node_id})-[:{type.value}]->(:val {{value: "{value}"}})'
        elif type == MemAssistInfoType.LOC:
            query_create_location, location_node_id, _ = QueryBuilder.query_create_noun_phrase(value)
            query += query_create_location
            query += f' create ({node_id})-[:{type.value}]->({location_node_id})'
        elif type in [MemAssistInfoType.TIME_POINT, MemAssistInfoType.TIME_START, MemAssistInfoType.TIME_END,
                      MemAssistInfoType.TIME_RANGE, MemAssistInfoType.TIME_DURATION]:
            query += f' create ({node_id})-[:{type.value}]->(:time {{value: "{value}"}})'

        print(query)
        result = execute_sparql_query(query)

    def get_value(self, entity, type=MemAssistInfoType.VAL):
        """ Get a detail of an entity from the database. """

        query, node_id = QueryBuilder.query_match_noun_phrase(entity)
        query += f' match ({node_id})-[:{type.value}]->(val) return val, {node_id} as entity'

        print(query)
        result = execute_sparql_query(query)
        values = [[record['val']['value'], record['entity']['value']] for record in result]

        return self.__prettyfy_result(values)

    def store_action(self, components):
        """
        Store a complete action of a subject, eventually together with other semantic entities
        (like location, timestamp, direct object, etc.).
        """

        query, subj_node_id, _ = QueryBuilder.query_create_noun_phrase(components['subj'], match_existing=True)

        query += f' create ({subj_node_id})-[:ACTION]->(act:action {{value: "{components["action"]}"}})'

        if components['ce']:
            sub_query, node_id, _ = QueryBuilder.query_create_noun_phrase(components['ce'])
            query += sub_query
            query += f' create (act)-[:CE]->({node_id})'

        if components['loc']:
            for loc in components['loc']:
                query_create_location, location_node_id, _ = QueryBuilder.query_create_noun_phrase(loc)
                query += query_create_location
                query += f' create (act)-[:LOC]->({location_node_id})'

        if components['time']:
            for i, time in enumerate(components['time']):
                query += f' create (act)-[:{time[1].value}]->(t{i}:time {{value: "{time[0]}"}})'

        print(query)
        result = execute_sparql_query(query)

    def get_action_time(self, components, info_type=MemAssistInfoType.TIME_POINT):
        """
        Get timestamp of an action expressed through a complex sentence
        (containing more than the entity whose time is requested).
        """

        # try to find the subject of the action
        query, subj_node_id = QueryBuilder.query_match_noun_phrase(components['subj'])

        # match the requested action
        query += f' match ({subj_node_id})-[:ACTION]->(act {{value: "{components["action"]}"}})'

        noun_phrase_nodes = [subj_node_id]
        if components['ce']:
            sub_query, node_id = QueryBuilder.query_match_noun_phrase(components['ce'])
            query += sub_query
            query += f' match (act)-[:CE]->({node_id})'
            noun_phrase_nodes.append(node_id)

        if components['loc']:
            for loc in components['loc']:
                query_create_location, location_node_id = QueryBuilder.query_match_noun_phrase(loc)
                query += query_create_location
                query += f' match (act)-[:LOC]->({location_node_id})'
                noun_phrase_nodes.append(location_node_id)

        if components['time']:
            for i, time in enumerate(components['time']):
                query += f' match (act)-[:{time[1].value}]->(t {{value: "{time[0]}"}})'

        # extract the requested property of the action
        query += f' match (act)-[:{info_type.value}]->(time) return time'
        if noun_phrase_nodes:
            query += ', ' + ', '.join(noun_phrase_nodes)
        print(query)
        result = execute_sparql_query(query)
        values = [[record['time']['value']] + [record[np]['value'] for np in noun_phrase_nodes] for record in result]

        return self.__prettyfy_result(values)

    @staticmethod
    def __prettyfy_result(values):
        if not values:
            return "Nu știu"
        if len(values) == 1:
            return values[0][0]
        return '\n'.join([f"▪ {', '.join(val[1:])}: ➜ {val[0]}" for val in values])
