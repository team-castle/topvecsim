import os
import time
import asyncio
import numpy as np
import streamlit as st
from top2vec import Top2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
os.environ[
    "REDIS_OM_URL"
] = "redis://default:h6zXURWzS3rYWENlzqwXkvI7dTJGrupO@redis-18366.c21900.ap-south-1-1.ec2.cloud.rlrcp.com:18366"

from topvecsim.models import Paper
from topvecsim.redis_utils import redis_conn
from topvecsim.search_index import SearchIndex

search_index = SearchIndex()


@st.experimental_memo
def load_model():
    model = Top2Vec.load("quick.top2vec")
    return model


def tokenize(doc):
    return simple_preprocess(strip_tags(doc), deacc=True)


st.title("Query Documents using Keywords")
st.write(
    "This Proof of Concept uses a Top2Vec model to encode your vectors. It then "
    "performs a similarity search on Redis to find the k-nearest neighbours."
)

title = st.text_input("Search Query", "Euclidean distance")

model = load_model()

num_papers = st.slider("How many papers would you like to see?", 0, 50, 25)
f_button = st.button("Fetch Papers")

if "f_button" not in st.session_state:
    st.session_state["f_button"] = False

if f_button:
    st.session_state["f_button"] = not st.session_state["f_button"]
    if st.session_state["f_button"]:
        docs = []
        with st.spinner("Fetching papers from Redis"):
            vec = model.model.infer_vector(
                doc_words=tokenize(title), alpha=0.025, min_alpha=0.01, epochs=100
            )

            query = search_index.vector_query(
                categories=[], years=[], number_of_results=30
            )

            try:
                res = asyncio.run(
                    redis_conn.ft("papers").search(
                        query,
                        query_params={
                            "vec_param": np.array(vec, dtype=np.float32).tobytes()
                        },
                    )
                )

            except Exception as e:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = asyncio.run(
                    redis_conn.ft("papers").search(
                        query,
                        query_params={
                            "vec_param": np.array(vec, dtype=np.float32).tobytes()
                        },
                    )
                )

            for doc in res.docs[:num_papers]:
                try:
                    p = asyncio.run(Paper.find(Paper.paper_id == doc.paper_id).first())
                except Exception as e:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    p = asyncio.run(Paper.find(Paper.paper_id == doc.paper_id).first())

                docs.append(
                    {
                        "similarity_percentage": round(
                            float(doc.vector_score) * 100, 2
                        ),
                        "paper_id": doc.paper_id,
                        "paper_pk": doc.paper_pk,
                        "container": st.container(),
                        "title": p.title,
                        "abstract": p.abstract,
                    }
                )

        if docs:
            for doc in docs:
                label = (
                    str(doc["title"])
                    + "  |  "
                    + str(doc["similarity_percentage"])
                    + "%"
                )
                with st.expander(label):
                    st.write(doc["abstract"])
                    st.markdown(
                        f"[ArXiv Paper Link](https://arxiv.org/abs/{doc['paper_id']})"
                    )
