FROM python:3.8-slim-buster

RUN pip3 install rasa

WORKDIR /opt/pepper/

COPY . .

EXPOSE 5005

CMD ["rasa", "run", "--enable-api", "--cors *"]