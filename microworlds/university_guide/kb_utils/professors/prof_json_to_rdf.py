import json
import codecs
import re
from unidecode import unidecode
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF


#########################################################################################################
# JS code used to parse the data from https://cs.pub.ro/index.php/?option=com_comprofiler&task=userslist

# for (let tr of $0.children)
#     console.log(`{"name": "${tr.children[1].innerText}", "position": "${tr.children[2].innerText}",
#     "office": "${tr.children[3].innerText}"}`);
#########################################################################################################

def same_person(name1, name2):
    """ Check if two names refer to the same person. """

    # Remove diacritics
    name1 = unidecode(name1)
    name2 = unidecode(name2)

    tokens1 = re.split(' |-', name1.lower())
    tokens2 = re.split(' |-', name2.lower())

    intersection = [token for token in tokens1 if token in tokens2]
    return len(intersection) >= 2  # at least 2 name tokens are the same


def clean(string):
    return re.sub(r'\s+', ' ', string.strip())


def serialize_rdf_graph(rdf_graph):
    """ Export the RDF triples as a Turtle file (.ttl). """

    with codecs.open("../../kb/professors.ttl", "w", "utf-8") as rdf_file:
        # Add a description of the data
        rdf_file.write('\n'.join([
            '# This file contains the RDF representation of data about teachers from',
            '# Faculty of Automatic Control and Computers, University Politehnica of Bucharest',
            '',
            '# NOTE: This file was autogenerated'
        ]))
        rdf_file.write('\n\n')

        rdf_file.write(rdf_graph.serialize(format='turtle').decode("utf-8"))


def main():
    # Initialize RDF graph
    rdf_graph = Graph()

    # Define prefixes
    base = 'http://www.readerbench.com/pepper#'
    rdf_graph.bind('', base)
    _ = Namespace(base)
    rdf_graph.bind('', _)

    # Load information about teachers
    f_cs50 = codecs.open('prof-50.cs.pub.ro.json', 'r', 'utf-8')
    cs50 = json.load(f_cs50)

    f_cs = codecs.open('prof-cs.pub.ro.json', 'r', 'utf-8')
    cs = json.load(f_cs)

    for prof in cs50:
        # Find the teacher in the cs.pub.ro list
        prof_cs = [entry for entry in cs if same_person(prof['name'], entry['name'])]
        prof_cs = prof_cs[0] if prof_cs else {}

        office = re.sub(r'\s|\.|-', '', prof_cs.get("office", ""))
        if not re.search(r'[\w0-9]+', office):
            # No valid office
            office = ""

        professor = URIRef(base + clean(prof["name"]).replace(" ", "_"))
        rdf_graph.add((professor, _.name, Literal(clean(prof["name"]))))
        rdf_graph.add((professor, _.description, Literal(clean(prof.get("description", "")), lang='en')))
        rdf_graph.add((professor, _.description, Literal(clean(prof.get("descriptionRo", "")), lang='ro')))
        rdf_graph.add((professor, _.coord, Literal(clean(prof.get("coord", "")), lang='en')))
        rdf_graph.add((professor, _.coord, Literal(clean(prof.get("coordRo", "")), lang='ro')))
        rdf_graph.add((professor, _.office, Literal(office)))
        rdf_graph.add((professor, RDF.type, _.Professor))

    serialize_rdf_graph(rdf_graph)


if __name__ == "__main__":
    main()
