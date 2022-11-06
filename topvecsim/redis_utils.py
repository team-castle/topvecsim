import os
import asyncio
import numpy as np
from typing import List, Dict, Any
from redis.commands.search.field import TagField
from aredis_om import get_redis_connection, Migrator

from topvecsim.models import Paper
from topvecsim.search_index import SearchIndex

# `redis_conn` can be imported from this script to access non-OM Redis methods.
redis_conn = get_redis_connection(
    url=os.getenv(
        "REDIS_OM_URL",
        f"redis://default:{os.getenv('REDIS_PASSWORD')}@redis-18891.c21900.ap-south-1-1.ec2.cloud.rlrcp.com:18891/0",
    ),
    decode_responses=False,
)


async def setup_indexes():
    """Run the migrations necessary to setup the indexes on Redis."""
    await Migrator().run()


async def gather_with_concurrency(
    papers: List[Dict[str, Any]], vectors: List[np.ndarray], sem_counter: int
):
    semaphore = asyncio.Semaphore(sem_counter)

    async def load_paper(paper, vec):
        async with semaphore:
            paper["paper_id"] = paper.pop("id")
            paper["categories"] = paper["categories"].replace(",", "|")
            paper_instance = Paper(**paper)

            await paper_instance.save()

            key = f"paper_vector:{str(paper_instance.paper_id)}"

            await redis_conn.hset(
                key,
                mapping={
                    "paper_pk": paper_instance.pk,
                    "paper_id": paper_instance.paper_id,
                    "categories": paper_instance.categories,
                    "year": paper_instance.year,
                    "vector": np.array(vec, dtype=np.float32).tobytes(),
                },
            )

    # Load papers concurrently.
    await asyncio.gather(*[load_paper(p, v) for p, v in zip(papers, vectors)])


async def load_data(
    papers: List[Dict[str, Any]], vectors: List[np.ndarray], sem_counter: int
):
    """Load the paper metadata and the vectors into Redis."""

    await gather_with_concurrency(
        papers,
        vectors,
        sem_counter=sem_counter,
    )

    await setup_vector_index(len(papers))


async def setup_vector_index(number_of_vectors: int, prefix: str = "paper_vector:"):
    """Setup the vector index."""

    # Run the migration to setup the indexes.
    await setup_indexes()

    search_index = SearchIndex()

    # Setup the index.
    categories_field = TagField("categories", separator="|")
    year_field = TagField("year", separator="|")

    await search_index.create_hnsw(
        categories_field,
        year_field,
        redis_conn=redis_conn,
        number_of_vectors=number_of_vectors,
        prefix=prefix,
        distance_metric="IP",
    )
