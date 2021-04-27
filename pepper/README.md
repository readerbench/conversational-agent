# Pepper conversational agent

## Training the agent with multiple microworlds

```shell
rasa train --data microworlds/generic/data microworlds/mem_assistant/data microworlds/university_guide/data
```

## Running the agent

#### Starting Pepper

1. Start actions HTTP server: `rasa run actions`
2. Start a RASA shell to communicate with the agent from the command line: `rasa shell`

#### Generating lookup tables

```shell
python3 grakn_lookup.py
```

### GraphDB container

The knowledge base is implemented using a GraphDB graph database (based on [RDF](https://www.w3.org/RDF/) triples).

#### Running the GraphDB container

```shell
docker run -p7200:7200 -v graphdb:/opt/graphdb-instance --name graphdb graphdb
```

#### Programmatically creating a new GraphDB repository

```shell
curl -X POST --header "Content-Type:multipart/form-data" -F "config=@/opt/graphdb-free-9.4.1/configs/config.ttl" "http://localhost:7200/rest/repositories"
```

## Deployment

Build all docker images
```shell
docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/pepper-web -f ../pepper-web-frontend/Dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/pepper-web

docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-server -f Dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-server

docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-actions  -f actions/Dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-actions

docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/graphdb -f graphdb.dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/graphdb
```