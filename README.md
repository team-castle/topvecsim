![logo](img/topvecsim.png)

# TopVecSim

This library was built for the 2022 Vector Search Hackathon hosted by the MLOps
Community, Redis, and Saturn Cloud.

Here, you'll find code and techniques that leverage the power of Redisearch to expose
topic modelling capabilities in a distributed manner. By storing the vectors remotely
in a performant Redis DB, we usher in the possibility of a scalable system where vector
addition, maintenance, and access can be taken care of by completely independent
processes.


# Attribution

This project is only possible because it did not have a cold start. We borrowed from
and read through code from these wonderful open source repositories:

### Redis arXiv Search

Link: https://github.com/RedisVentures/redis-arXiv-search

### Redis Product Search

Link: https://github.com/RedisVentures/redis-product-search/

### Redis Python

Link: https://github.com/redis/redis-py

### Redis OM Python

Link: https://github.com/redis/redis-om-python

### Top2Vec

Link: https://github.com/ddangelov/Top2Vec/


# Quick Start

pip install .

# Set up this repository for development

## Setup Poetry

Install Poetry:

`curl -sSL https://install.python-poetry.org | python3 - --preview`

## Create a new environment

`poetry shell`

This automatically creates a new virtual environment if there isn't one already.

## Install Dependencies

`poetry install`
