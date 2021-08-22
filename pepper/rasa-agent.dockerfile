FROM python:3.8-slim-buster

RUN pip3 install rasa==2.0.3
RUN pip3 install spacy==2.2.3

WORKDIR /opt/pepper/

COPY . .

CMD rasa run -p 80 --model models --enable-api --endpoints endpoints.compose.yml --cors *