FROM python:3.8-slim-buster

RUN pip3 install rasa unidecode requests googletrans

WORKDIR /opt/pepper/

COPY . actions/

#RUN pip3 install --no-cache-dir -r actions/requirements.txt

EXPOSE 5055

CMD ["rasa", "run", "actions"]