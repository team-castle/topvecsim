import numpy as np
import pandas as pd
from umap import UMAP
import umap.umap_ as umap
from top2vec import Top2Vec
from joblib import dump, load
from typing import List, Literal, Optional

from topvecsim.data import get_df


def train_top2vec_model(
    documents: List[str],
    speed: Literal["fast-learn", "learn", "deep-learn"] = "learn",
    workers: int = 16,
    embedding_model: str = "doc2vec",
    use_corpus_file: bool = True,
    **kwargs,
):
    """Train a Top2Vec model with the provided list of documents and arguments to tune
    the model.

    All these parameters are directly passed to the Top2Vec model, so for a detailed
    explanation, it's best to check out the top2vec Github Repository.

    Parameters
    ----------
    documents : list of strings
        The documents used to train the model.
    speed : string
        The mode in which the model will be trained.

        fast-learn -> 4 epochs
        learn -> 40 epochs
        deep-learn -> 400 epochs
    workers : int
        Make this equal to the number of CPUs your machine possesses.
    embedding_model : str
        This is doc2vec by default. Best to leave it at this since this is where we
        got the best results even if the trainings took time; this is because the model
        is trained from scratch.
    kwargs : dict
        Any arguments you want to pass directly to the Top2Vec class as mentioned in
        their documentation.
    """

    model = Top2Vec(
        documents=documents,
        speed=speed,
        workers=workers,
        embedding_model=embedding_model,
        use_corpus_file=use_corpus_file,
        **kwargs,
    )

    print("Training complete.")

    return model


def save_top2vec_model(model: Top2Vec, save_path: str) -> None:
    """Save the Top2Vec to the provided path.

    Parameters
    ----------
    model : Top2Vec
        The trained model that will be saved.
    save_path : string
        The path to which the model will be saved.
    """

    print(f"Saving Top2Vec model to {save_path}")

    model.save(save_path)


def load_top2vec_model(model_path: str) -> Top2Vec:
    """Given the model path, load the Top2Vec model.

    Parameters
    ----------
    model_path : string
        The path to the saved Top2Vec model.

    Returns
    -------
    Top2Vec
        The loaded model.
    """

    print(f"Loading Top2Vec model from {model_path}")

    return Top2Vec.load(model_path)


def train_umap_model(vectors: np.ndarray, **kwargs) -> UMAP:
    """Instantiate and train a UMAP model with the provided vectors.

    Parameters
    ----------
    vectors : numpy array
        The vectors on which to train the UMAP model.

    Returns
    -------
    UMAP
        The trained UMAP model.
    """

    model = umap.UMAP(**kwargs)

    model.fit(vectors)

    return model


def save_umap_model(
    model: UMAP,
    save_path: str,
) -> None:
    dump(model, save_path)


def load_umap_model(model_path: str) -> UMAP:
    return load(model_path)


def load_train_save_umap(
    top2vec_model_path: str, umap_save_path: str, **umap_kwargs
) -> None:
    """Load a Top2Vec model from file and train a UMAP model with the retrieved
    document vectors.

    Parameters
    ----------
    top2vec_model_path : string
        The path from which the Top2Vec model will be loaded.
    umap_save_path : string
        The path to which the UMAP model will be saved.
    umap_kwargs : dict
        The keyword arguments used to configure the UMAP model that will be trained.
    """

    if not umap_kwargs:
        umap_kwargs = {
            "n_neighbors": 50,
            "n_components": 2,  # 5 -> 2 for plotting
            "metric": "cosine",
        }

    top2vec_model = load_top2vec_model(top2vec_model_path)

    umap_model = train_umap_model(top2vec_model.document_vectors, **umap_kwargs)

    save_umap_model(umap_model, umap_save_path)


def train_save_top2vec(
    save_path: str,
    column_name: str = "input",
    data_file: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    workers: int = 8,
    speed: Literal["fast-learn", "learn", "deep-learn"] = "learn",
    embedding_model: str = "doc2vec",
    **top2vec_kwargs,
):
    """Process the JSON dataset, train a Top2Vec model, and save it to the provided
    path.

    Parameters
    ----------
    save_path : string
        The path to which the trained Top2Vec model will be saved.
    column_name : string
        The column in the dataset that contains the strings that will be used to train
        the model.
    data_file : optional string
        The path to the Dataset in JSON form.
    df : optional pandas dataframe
        The dataset as a dataframe.
    workers : int
        Make this equal to the number of CPUs your machine possesses.
    speed : string
        The mode in which the model will be trained.

        fast-learn -> 4 epochs
        learn -> 40 epochs  # Best reasonable option
        deep-learn -> 400 epochs
    embedding_model : str
        This is "doc2vec" by default. Best to leave it at this since this is where we
        got the best results even if the trainings took time; this is because the model
        is trained from scratch.
    """

    assert (
        df is not None
    ) or data_file, "One of `df` or `data_file` has to be available"

    if df is None:
        assert isinstance(data_file, str), "data_file has to be a string."

        print(f"Generating a Pandas Dataframe from: {data_file}")

        df = get_df(data_file=data_file)

        print(f"Created Dataframe of shape: {df.shape}")
    else:
        assert isinstance(df, pd.DataFrame)

    model = train_top2vec_model(
        documents=df[column_name].tolist(),
        workers=workers,
        speed=speed,
        embedding_model=embedding_model,
        **top2vec_kwargs,
    )

    save_top2vec_model(model=model, save_path=save_path)

    return model
