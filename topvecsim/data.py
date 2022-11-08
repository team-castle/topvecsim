import re
import json
import pickle
import string
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Iterator

from topvecsim import logger
from topvecsim.minio_utils import minio_client

YEAR_PATTERN = r"(19|20[0-9]{2})"


def clean_description(description: str) -> str:
    """Clean the description string.

    Parameters
    ---------
    description : string
        The description of the paper.

    Returns
    -------
    string
        The cleaned description.
    """

    # Handle case where description is not present.
    if not description:
        return ""

    # Remove Unicode characters.
    description = description.encode("ascii", "ignore").decode()

    # Remove punctuation.
    description = re.sub("[%s]" % re.escape(string.punctuation), " ", description)

    # Clean up the spacing.
    description = re.sub("\s{2,}", " ", description)

    # Remove urls.
    # description = re.sub("https*\S+", " ", description)

    # Remove newlines.
    description = description.replace("\n", " ")

    # Remove all numbers.
    # description = re.sub('\w*\d+\w*', '', description)

    # Split on capitalized words.
    description = " ".join(re.split("(?=[A-Z])", description))

    # Clean up the spacing again.
    description = re.sub("\s{2,}", " ", description)

    # Make all words lowercase.
    description = description.lower()

    return description


def process(paper_json: str):
    """Takes in the record in string form, converts it to dict, then cleans it up and
    returns it.

    Parameters
    ----------
    paper_json : str
        The paper info in JSON string form.

    Returns
    -------
    The paper info as a dictionary.
    """

    paper: Dict[str, Union[str, int]] = json.loads(paper_json)

    # Parse the date with RegEx.
    if paper["journal-ref"]:
        years = [int(year) for year in re.findall(YEAR_PATTERN, paper["journal-ref"])]
        years = [year for year in years if (year <= 2022 and year >= 1991)]
        year = min(years) if years else None
    else:
        year = None

    return {
        "id": paper["id"],
        "title": paper["title"],
        "year": year,
        "authors": paper["authors"],
        "categories": ",".join(paper["categories"].split(" ")),
        "abstract": paper["abstract"],
        "input": clean_description(f"{paper['title']} {paper['abstract']}"),
    }


def papers(
    data_file: str, cat_filter: Union[str, List[str]]
) -> Iterator[Dict[str, Any]]:
    """Get a path to a JSON file, iterate through the records and yield them.

    Parameters
    ----------
    data_file : str
        The path to the dataset JSON file.

    Returns
    -------
    An iterator that yield records from the dataset file.
    """

    with open(data_file, "r") as f:
        for paper in f:
            paper = process(paper)
            if isinstance(cat_filter, str):
                if paper["categories"].startswith(cat_filter):
                    yield paper
            elif isinstance(cat_filter, list):
                if any(sub_cat in paper["categories"] for sub_cat in cat_filter):
                    yield paper


def get_df(
    data_file: str = "/home/jovyan/arxiv/arxiv-metadata-oai-snapshot.json",
    cat_filter: Union[str, List[str]] = "cs.",
) -> pd.DataFrame:
    """Create and return a Pandas DataFrame from a file path.

    Parameters
    ----------
    data_file : str
        The path to the Dataset in JSON form.
    cat_filter : string or list of strings
        Category filter to limit the data from the arXiv dataset that will be
        considered for the training.
    """

    df = pd.DataFrame(list(papers(data_file, cat_filter)))

    logger.info(f"Created DataFrame of shape {df.shape}, here's a sample:\n")
    logger.info(df.head())

    return df


def save_df_as_pkl(df: pd.DataFrame, save_path: str, key: Optional[str] = None):
    """Save Dataframe to a Pickle file.

    Parameters
    ----------
    df : pandas dataframe
        The dataframe that will be saved to disk.
    """

    with open(save_path, "wb") as f:
        pickle.dump(df, f)

    if key and minio_client:
        minio_client.upload_from_path_to_key(save_path, key)


def load_df_from_pkl(save_path: str) -> pd.DataFrame:
    """Load Dataframe from a Pickle file and return it."""

    with open(save_path, "rb") as f:
        df = pickle.load(f)

    return df
