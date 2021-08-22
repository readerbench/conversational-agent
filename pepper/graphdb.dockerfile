FROM openjdk:16.0.2-buster

COPY graphdb/graphdb-free-9.9.0 /opt/graphdb-free-9.9.0/

# Copy config file used to create a new GraphDB repository
COPY graphdb/config.ttl /opt/graphdb-free-9.9.0/configs/config.ttl

ENV PATH="/opt/graphdb-free-9.9.0/bin:${PATH}"

#RUN apk add --no-cache --upgrade bash
#RUN apk --no-cache add curl
#
## Install OpenSSH and set the password for root. In this example, "apk add" is the install instruction for an Alpine Linux-based image.
#RUN apk add openssh && echo "root:Rpivsw3" | chpasswd

## Copy the sshd_config file to the /etc/ssh/ directory
#COPY graphdb/sshd_config /etc/ssh/

WORKDIR /opt/graphdb-free-9.9.0

COPY ./microworlds/university_guide/data/kb ./kb-files/

## Open port 2222 for SSH access
#EXPOSE $PORT 2222

ENV PORT=80

CMD cat ./configs/config.ttl && \
    loadrdf -c ./configs/config.ttl -m serial ./kb-files/* && \
    until graphdb -Dgraphdb.page.cache.size=200m -Dpool.buffer.size=20000 -Dinfer.pool.size=1 -Dgraphdb.connector.port=$PORT; do free; echo -e "\n\nDB crashed with exit code $?. Respawning..\n\n"; sleep 5; done