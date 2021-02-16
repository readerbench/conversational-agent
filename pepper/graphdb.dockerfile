FROM openjdk:16-alpine3.12

COPY graphdb/graphdb-free-9.4.1 /opt/graphdb-free-9.4.1/

# Copy config file used to create a new GraphDB repository
COPY graphdb/config.ttl /opt/graphdb-free-9.4.1/configs/config.ttl

ENV PATH="/opt/graphdb-free-9.4.1/bin:${PATH}"

RUN apk add --no-cache --upgrade bash
RUN apk --no-cache add curl

WORKDIR /opt/graphdb-free-9.4.1

COPY ./microworlds/university_guide/data/kb ./kb-files/

ENV PORT=7200

EXPOSE $PORT

CMD ./bin/loadrdf -c ./configs/config.ttl -m parallel ./kb-files/* &&
    graphdb -Dgraphdb.home=/opt/graphdb-instance -Dgraphdb.connector.port=$PORT