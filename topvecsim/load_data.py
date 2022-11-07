from top2vec import Top2Vec
from typing import List, Dict, Any, Optional

from topvecsim.redis_utils import (
    load_papers,
    load_words,
    load_topics,
    setup_vector_index,
)


async def load_data(
    documents: Dict[str, Any],
    vocab: Dict[str, Any],
    topics: Dict[str, Any],
    num_topic_words_to_store: int = 10,
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
        num_topic_words_to_store=num_topic_words_to_store,
        sem_counter=sem_counter,
    )

    await setup_vector_index(
        index_name="papers",
        number_of_vectors=len(document_data),
        prefix="paper_vector:",
    )

    await setup_vector_index(
        index_name="words",
        number_of_vectors=len(words),
        prefix="word_vector:",
    )

    await setup_vector_index(
        index_name="topics",
        number_of_vectors=len(topic_words),
        prefix="topic_vector:",
    )


async def load_all_data(
    papers: List[Dict[str, Any]],
    model: Top2Vec,
    num_topic_words_to_store: int = 15,
    sem_counter: int = 20,
) -> None:
    """Extract information from papers and the Top2Vec model to pass into `load_data`.

    Parameters
    ----------
    papers : list of dicts
        The paper metadata. The data frame as records.
    model : Top2Vec
        The trained model from which the relevant vectors will be extracted.
    num_topic_words_to_store : int
        Controls the number of words assigned to each topic.
    sem_counter : int
        Controls concurrency. Use this to limit the number of parallel clients
        connected to your Redis DB.
    """
    await load_data(
        documents={
            "data": papers,
            "indexes": list(range(len(papers))),
            "vectors": model.document_vectors,
            "topics": model.doc_top,
            "topic_scores": model.doc_dist,
        },
        vocab={
            "words": model.vocab,
            "indexes": list(range(len(model.vocab))),
            "vectors": model.word_vectors,
        },
        topics={
            "indexes": list(range(len(model.topic_vectors))),
            "vectors": model.topic_vectors,
            "sizes": model.topic_sizes.tolist(),
            "words": model.topic_words,
            "word_scores": model.topic_word_scores,
        },
        num_topic_words_to_store=num_topic_words_to_store,
        sem_counter=sem_counter,
    )
