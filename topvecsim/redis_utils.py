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
    num_words_to_store: int,
    sem_counter: int,
):
    semaphore = asyncio.Semaphore(sem_counter)

    async def load_topic(words, word_scores, idx, size, vec):
        async with semaphore:
            key = f"topic_vector:{idx}"

            # Since Redis Hashes can't contain lists, we add a key for each topic word.
            mapping = {
                f"word_{i}": v for i, v, in enumerate(words[:num_words_to_store])
            }
            mapping.update(
                {
                    f"word_score_{i}": float(v)
                    for i, v, in enumerate(word_scores[:num_words_to_store])
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


async def load_data(
    documents: Dict[str, Any],
    vocab: Dict[str, Any],
    topics: Dict[str, Any],
    num_words_to_store: int = 10,
    sem_counter: int = 16,
):
    """Load the paper metadata and the vectors into Redis."""

    document_data = documents["data"]  # The actual content of each document.
    document_indexes = documents["indexes"]  # The position of the document.
    document_vectors = documents["vectors"]
    document_topics = documents["topics"]  # Topic IDX for each document.
    document_topic_scores = documents["topic_scores"]  # Scores for each document topic.

    words = vocab["words"]
    word_indexes = vocab["indexes"]
    word_vectors = vocab["vectors"]

    topic_indexes = topics["indexes"]  # The position of each topic in the topic list.
    topic_vectors = topics["vectors"]
    topic_sizes = topics["sizes"]  # Frequency of documents per topic.
    topic_words = topics["words"]  # 2D array of 50 words to describe every topic.
    topic_word_scores = topics["word_scores"]
    """2D array of 50 scores to show the distance between every word assigned to a
    topic and the topic itself."""

    await load_papers(
        papers=document_data,
        indexes=document_indexes,
        topics=document_topics,
        topic_scores=document_topic_scores,
        vectors=document_vectors,
        sem_counter=sem_counter,
    )

    await load_words(
        words=words,
        word_indexes=word_indexes,
        sem_counter=sem_counter,
        word_vectors=word_vectors,
    )

    await load_topics(
        topic_words=topic_words,
        topic_word_scores=topic_word_scores,
        topic_indexes=topic_indexes,
        topic_sizes=topic_sizes,
        topic_vectors=topic_vectors,
        num_words_to_store=num_words_to_store,
        sem_counter=sem_counter,
    )

    # await setup_vector_index(number_of_vectors=len(document_data))


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
