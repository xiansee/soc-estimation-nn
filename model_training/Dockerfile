FROM python:3.11-slim

RUN apt-get update
RUN apt-get upgrade

WORKDIR /app

ADD https://github.com/xiansee/model-trainer/releases/download/v1.0.0/model-trainer-1.0.0.tar.gz dist/model-trainer-1.0.0.tar.gz
RUN pip install dist/*.tar.gz
