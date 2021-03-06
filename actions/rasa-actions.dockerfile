FROM python:3.8-slim-buster

RUN pip3 install rasa~=3.0.9
RUN pip3 install unidecode requests

WORKDIR /opt/pepper/

COPY . actions/

WORKDIR /opt/pepper/actions

EXPOSE 80

CMD rasa run actions -p 80