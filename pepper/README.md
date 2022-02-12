# Pepper conversational agent

## Training the agent with multiple microworlds

```shell
rasa train --data \
  microworlds/generic/data \
  microworlds/mem_assistant/data \
  microworlds/university_guide/data
```

## Running the agent

#### Starting Pepper

1. Start actions HTTP server: `rasa run actions`
2. Start a RASA shell to communicate with the agent from the command line: `rasa shell`
   or start the REST API: `rasa run -p 80 --enable-api --cors *`

#### [Deprecated] Generating lookup tables

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

### Build docker images

```shell
docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/pepper-web -f ../pepper-web-frontend/web-ui.prod.dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/pepper-web

docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-server -f rasa-agent.dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-server

docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-actions -f actions/rasa-actions.dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/rasa-actions

docker build -t registry.gitlab.com/gabrielboroghina/pepper-conv-agent/graphdb -f graphdb.dockerfile .
docker push registry.gitlab.com/gabrielboroghina/pepper-conv-agent/graphdb
```

### Setup nodes

1. Allow access to the Docker images registry (get deploy token e.g. from GitLab project's settings -> Repository ->
   Deploy tokens):

```shell
docker login -u <username> -p <deploy_token> registry.gitlab.com # Needs GitLab deploy token
```

2. Copy deployment files to the remote machine(s): `stack.yml` and `kong.yml`

3. Pull images and deploy the service stack:

```shell
docker swarm init # Initialize swarm of nodes
docker stack deploy -c stack.yml --with-registry-auth pepper # Start service stack

docker stack rm pepper # Stop the service stack
docker service ls # Check running services
docker container ls # Check running containers
docker service logs pepper_kong # Show logs from the kong service
```
