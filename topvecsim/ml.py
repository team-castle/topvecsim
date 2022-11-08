import os
import uuid
import mlflow
import numpy as np
import pandas as pd
from umap import UMAP
from pathlib import Path
import umap.umap_ as umap
from top2vec import Top2Vec
from joblib import dump, load
from typing import List, Dict, Literal, Optional, Union

from topvecsim import logger
from topvecsim.minio_utils import minio_client
from topvecsim.data import get_df, load_df_from_pkl, save_df_as_pkl

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def save_top2vec_model(
    model: Top2Vec, save_path: str, key: Optional[str] = None
) -> None:
    """Save the Top2Vec to the provided path.

    Parameters
    ----------
    model : Top2Vec
        The trained model that will be saved.
    save_path : string
        The path to which the model will be saved.
    key : string
        The remote key to which the file will be uploaded.
    """

    model.save(save_path)

    if key and minio_client:
        minio_client.upload_from_path_to_key(
            path=save_path,
            key=key,
        )


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

    logger.info(f"Loading Top2Vec model from {model_path}")

    return Top2Vec.load(model_path)


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

    logger.info("Loading Top2Vec model.")

    top2vec_model = load_top2vec_model(top2vec_model_path)

    logger.info("Beginning UMAP Training.")

    # Instantiate the UMAP model.
    model = umap.UMAP(**umap_kwargs)

    # Train the UMAP model with the document vectors.
    model.fit(top2vec_model.document_vectors)

    logger.info("Completed UMAP Training.")

    # Save the UMAP model to disk.
    save_umap_model(model, umap_save_path)

    logger.info(f"Saved trained UMAP model to {umap_save_path}")


def train_save_top2vec(
    name: str,
    save_path: str,
    column_name: str = "input",
    data_file: Optional[str] = None,
    df_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    workers: int = 8,
    speed: Literal["fast-learn", "learn", "deep-learn"] = "learn",
    embedding_model: str = "doc2vec",
    cat_filter: Union[str, List[str]] = "cs.",
    save_training_data: bool = True,
    **top2vec_kwargs,
) -> Dict[str, Union[Top2Vec, pd.DataFrame]]:
    """Process the JSON dataset, train a Top2Vec model, and save it to the provided
    path.

    ### Entrypoint to Top2Vec model training.

    Parameters
    ----------
    name : string
        A name for the current run. Used for tracking purposes.
    save_path : string
        The path to which the trained Top2Vec model will be saved.
    column_name : string
        The column in the dataset that contains the strings that will be used to train
        the model.
    data_file : optional string
        The path to the Dataset in JSON form.
    df_path : optional string
        The cleaned Dataset. Dataframe saved as a pickle file.
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
    cat_filter : string or list of strings
        Category filter to limit the data from the arXiv dataset that will be
        considered for the training.
    save_training_data : boolean
        Whether the training data should be saved to disk. True by default.
    """

    run_name = f"run_{str(uuid.uuid4())[:4]}"
    run_dir = f"/tmp/topvecsim/{run_name}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    mlflow.start_run(run_name=run_name)

    mlflow.log_param(key="num_workers", value=workers)

    assert (
        (df is not None) or data_file or df_path
    ), "One of `df`, `df_path`, or `data_file` has to be available"

    # If a JSON file is provided, pass it to get_df, clean it and get the dataframe.
    if data_file:
        assert (
            isinstance(data_file, str) and Path(data_file).exists()
        ), "data_file has to be a string."

        logger.info(f"Generating a Pandas Dataframe from: {data_file}")

        df = get_df(data_file=data_file, cat_filter=cat_filter)

        logger.info(f"Created Dataframe of shape: {df.shape}")

    # If a path to a Dataframe pickle file is given, we assume that it's already been
    # cleaned, and we load it directly.
    elif df_path:
        logger.info(f"Loaded Dataframe from Pickle file: {df_path}")

        df = load_df_from_pkl(df_path)

    # At this point, the user has passed in a Datafram directly, in which case we do
    # nothing.
    else:
        assert isinstance(df, pd.DataFrame)

        logger.info(f"Received Dataframe of shape: {df.shape}")

    # Save the training dataframe as a Pickle file to disk for later use.
    if save_training_data:
        save_df_as_pkl(
            df=df,
            save_path=f"{run_dir}/training_data.pkl",
            key=f"runs/{run_name}/training_data.pkl",
        )

    mlflow.log_param(key="num_documents", value=len(df))

    logger.info(f"Saved the training data as a pickle file to: {'training_data.pkl'}")
    logger.info("Beginning the model training.")

    # Train the Top2Vec model.
    model = Top2Vec(
        documents=df[column_name].tolist(),
        speed=speed,
        workers=workers,
        embedding_model=embedding_model,
        use_corpus_file=True,  # Speeds up training by using a tmp file.
        **top2vec_kwargs,
    )

    logger.info("Training complete.")

    mlflow.log_params(
        {"num_words": len(model.vocab), "num_topics": len(model.topic_vectors)}
    )

    # Save the Top2Vec model to disk and remote object storage.
    save_top2vec_model(
        model,
        save_path=f"{run_dir}/model.top2vec",
        key=f"runs/{run_name}/model.top2vec",
    )

    logger.info(f"Saved Top2Vec model to {save_path}.")

    mlflow.end_run()

    return {"model": model, "df": df}
