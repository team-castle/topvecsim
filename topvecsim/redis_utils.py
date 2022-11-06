import os
import asyncio
import numpy as np
from typing import List, Dict, Any
from aredis_om import get_redis_connection, Migrator

from topvecsim.models import Paper

# `redis_conn` can be imported from this script to access non-OM Redis methods.
redis_conn = get_redis_connection(decode_responses=False)


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
