import pandas as pd
from top2vec import Top2Vec
from typing import List, Dict, Any, Optional

from topvecsim.data import load_df_from_pkl
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
    index: bool = True,
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

    if index:
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
    limit_data: Optional[Dict[str, Any]] = None,
    index: bool = True,
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
    limit_data : Dict[str, Any]
        Helps control the amount of data loaded into the DB.
    index : boolean
        Whether or not to index the data.
    """

    papers_lower = 0
    papers_upper = len(papers)

    words_lower = 0
    words_upper = len(model.vocab)

    topics_lower = 0
    topics_upper = len(model.topic_vectors)

    if limit_data:
        if limit_data.get("papers"):
            papers_lower = limit_data["papers"].get("lower", 0)
            papers_upper = limit_data["papers"].get("upper", len(papers))

        if limit_data.get("words"):
            words_lower = limit_data["words"].get("lower", 0)
            words_upper = limit_data["words"].get("upper", len(model.vocab))

        if limit_data.get("topics"):
            topics_lower = limit_data["topics"].get("lower", 0)
            topics_upper = limit_data["topics"].get("upper", len(model.topic_vectors))

    await load_data(
        documents={
            "data": papers[papers_lower:papers_upper],
            "indexes": list(range(len(papers[papers_lower:papers_upper]))),
            "vectors": model.document_vectors[papers_lower:papers_upper],
            "topics": model.doc_top[papers_lower:papers_upper],
            "topic_scores": model.doc_dist[papers_lower:papers_upper],
        },
        vocab={
            "words": model.vocab[words_lower:words_upper],
            "indexes": list(range(len(model.vocab[words_lower:words_upper]))),
            "vectors": model.word_vectors[words_lower:words_upper],
        },
        topics={
            "indexes": list(range(len(model.topic_vectors[topics_lower:topics_upper]))),
            "vectors": model.topic_vectors[topics_lower:topics_upper],
            "sizes": model.topic_sizes[topics_lower:topics_upper].tolist(),
            "words": model.topic_words[topics_lower:topics_upper],
            "word_scores": model.topic_word_scores[topics_lower:topics_upper],
        },
        num_topic_words_to_store=num_topic_words_to_store,
        sem_counter=sem_counter,
        index=index,
    )


async def load_all_data_from_disk(
    model_path: str,
    df_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    num_topic_words_to_store: int = 15,
    sem_counter: int = 20,
    limit_data: Optional[Dict[str, Any]] = None,
    index: bool = True,
) -> Dict[str, Any]:
    """Extract information from the CSV/dataframe and the Top2Vec model to pass into
    `load_all_data`.

    Parameters
    ----------
    df_path : optional string
        The paper metadata. Dataframe saved as a pickle file.
    csv_path : optional string
        The paper metadata. CSV file.
    model_path : string
        Path to the saved model from which the relevant vectors will be extracted.
    num_topic_words_to_store : int
        Controls the number of words assigned to each topic.
    sem_counter : int
        Controls concurrency. Use this to limit the number of parallel clients
        connected to your Redis DB.
    limit_data : Dict[str, Any]
        Helps control the amount of data loaded into the DB.
    index : boolean
        Whether or not to index the data once it's loaded into the DB.
    """

    assert df_path or csv_path, "One of df_path or csv_path must be provided."

    if df_path:
        df = load_df_from_pkl(df_path)
    else:
        assert csv_path
        df = pd.read_csv(csv_path)

    papers = df.to_dict("records")
    model = Top2Vec.load(model_path)

    await load_all_data(
        papers,
        model,
        num_topic_words_to_store=num_topic_words_to_store,
        sem_counter=sem_counter,
        limit_data=limit_data,
        index=index,
    )

    return {
        "model": model,
        "df": df,
        "papers": papers,
    }
