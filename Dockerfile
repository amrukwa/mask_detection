FROM python:3.8-slim-buster

EXPOSE 5000

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN pip install poetry
RUN python -m pip install -r requirements-dev.txt
