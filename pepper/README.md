# Pepper conversational agent

## Running instructions

#### Starting Pepper

1. Start actions HTTP server: `rasa run actions`
2. Start a RASA shell to communicate with the agent from the command line: `rasa shell`

#### Generating lookup tables

```shell script
python3 grakn_lookup.py
```

### GraphDB container

The knowledge base is implemented using a GraphDB graph database (based on [RDF](https://www.w3.org/RDF/) triples).

#### Running the GraphDB container

```shell script
docker run -p7200:7200 -v graphdb:/opt/graphdb-instance --name graphdb graphdb
```

#### Programatically creating a new GraphDB repository
```shell script
curl -X POST --header "Content-Type:multipart/form-data" -F "config=@/opt/graphdb-free-9.4.1/configs/config.ttl" "http://localhost:7200/rest/repositories"
```
