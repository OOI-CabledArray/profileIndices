FROM python:3.11-slim-bookworm

COPY ./ /tmp/profileIndices

RUN pip install /tmp/profileIndices