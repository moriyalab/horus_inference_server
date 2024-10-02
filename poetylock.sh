#!/bin/bash

cp /home/root/docker/pyproject.toml /app/pyproject.toml
cd /app
poetry lock --no-update
rm -rf /home/root/docker/poetry.lock
cp /app/poetry.lock /home/root/docker/poetry.lock