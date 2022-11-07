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


async def load_papers(
    papers: List[Dict[str, Any]],
    indexes: List[int],
    topics: List[str],
    topic_scores: List[float],
    vectors: List[np.ndarray],
    sem_counter: int,
):
    semaphore = asyncio.Semaphore(sem_counter)

    async def load_paper(paper, idx, vec, topic, topic_score):
        async with semaphore:
            paper["paper_id"] = paper.pop("id")
            paper["categories"] = paper["categories"].replace(",", "|")

            paper["topic"] = str(topic)
            paper["topic_score"] = float(topic_score)

            paper_instance = Paper(**paper)

            await paper_instance.save()

            key = f"paper_vector:{str(paper_instance.paper_id)}"

            await redis_conn.hset(
                key,
                mapping={
                    "paper_pk": paper_instance.pk,
                    "paper_id": paper_instance.paper_id,
                    "doc_idx": idx,
                    "categories": paper_instance.categories,
                    "year": paper_instance.year,
                    "vector": np.array(vec, dtype=np.float32).tobytes(),
                },
            )

    # Load papers concurrently into Redis.
    await asyncio.gather(
        *[
            load_paper(paper, idx, vector, topic, topic_score)
            for paper, idx, vector, topic, topic_score in zip(
                papers, indexes, vectors, topics, topic_scores
            )
        ]
    )


async def load_words(
    words: List[str],
    word_indexes: List[int],
    word_vectors: List[np.ndarray],
    sem_counter: int,
):
    semaphore = asyncio.Semaphore(sem_counter)

    async def load_word(word, idx, vec):
        async with semaphore:
            key = f"word_vector:{idx}"

            await redis_conn.hset(
                key,
                mapping={
                    "word_id": int(idx),
                    "word": str(word),
                    "vector": np.array(vec, dtype=np.float32).tobytes(),
                },
            )

    # Load papers concurrently into Redis.
    await asyncio.gather(
        *[
            load_word(word, idx, vector)
            for word, idx, vector in zip(words, word_indexes, word_vectors)
        ]
    )


async def load_topics(
    topic_words: List[np.ndarray],
    topic_word_scores: List[int],
    topic_indexes: List[int],
    topic_sizes: List[int],
    topic_vectors: List[np.ndarray],
    num_topic_words_to_store: int,
    sem_counter: int,
):
    """Load the topics into Redis asynchronously."""

    semaphore = asyncio.Semaphore(sem_counter)

    async def load_topic(words, word_scores, idx, size, vec):
        async with semaphore:
            key = f"topic_vector:{idx}"

            # Since Redis Hashes can't contain lists, we add a key for each topic word.
            mapping = {
                f"word_{i}": v for i, v, in enumerate(words[:num_topic_words_to_store])
            }
            mapping.update(
                {
                    f"word_score_{i}": float(v)
                    for i, v, in enumerate(word_scores[:num_topic_words_to_store])
                }
            )
            mapping.update(
                {
                    "word_id": int(idx),
                    "size": str(size),
                    "vector": np.array(vec, dtype=np.float32).tobytes(),
                }
            )

            await redis_conn.hset(
                key,
                mapping=mapping,
            )

    # Load papers concurrently into Redis.
    await asyncio.gather(
        *[
            load_topic(words, word_scores, idx, size, vec)
            for words, word_scores, idx, size, vec in zip(
                topic_words,
                topic_word_scores,
                topic_indexes,
                topic_sizes,
                topic_vectors,
            )
        ]
    )


async def setup_vector_index(index_name: str, number_of_vectors: int, prefix: str):
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
        index_name=index_name,
        redis_conn=redis_conn,
        number_of_vectors=number_of_vectors,
        prefix=prefix,
        distance_metric="IP",
    )
