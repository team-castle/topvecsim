import re
from redis.asyncio import Redis
from typing import Optional, Pattern
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


class TokenEscaper:
    """Escape punctuation within an input string. Taken from RedisOM Python."""

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/ ]"

    def __init__(self, escape_chars_re: Optional[Pattern] = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        def escape_symbol(match):
            value = match.group(0)
            return f"\\{value}"

        return self.escaped_chars_re.sub(escape_symbol, value)


class SearchIndex:
    """SearchIndex is used to wrap and capture all information and actions applied to a
    RediSearch index including creation, management, and query construction.
    """

    escaper = TokenEscaper()

    async def create_flat(
        self,
        *fields,
        index_name: str,
        redis_conn: Redis,
        number_of_vectors: int,
        prefix: str,
        distance_metric: str = "L2",
    ) -> None:
        """Create a FLAT aka brute force style index.

        Parameters
        ----------
        index_name : string
            The name of the index to be created.
        redis_conn : Redis
            Redis connection object.
        number_of_vectors : int
            Count of the number of initial vectors.
        prefix : string
            Key prefix to use for RediSearch index creation.
        distance_metric : optional string
            Distance metric to use for Vector Search. Defaults to 'L2'.
        """

        vector_field = VectorField(
            "vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": 300,
                "DISTANCE_METRIC": distance_metric,
                "INITIAL_CAP": number_of_vectors,
                "BLOCK_SIZE": number_of_vectors,
            },
        )

        await self._create(
            *fields,
            vector_field,
            index_name=index_name,
            redis_conn=redis_conn,
            prefix=prefix,
        )

    async def create_hnsw(
        self,
        *fields,
        index_name: str,
        redis_conn: Redis,
        number_of_vectors: int,
        prefix: str,
        distance_metric: str = "COSINE",
    ) -> None:
        """Create an approximate NN index via HNSW.

        Parameters
        ----------
        index_name : string
            The name of the index to be created.
        redis_conn : Redis
            Redis connection object.
        number_of_vectors : int
            Count of the number of initial vectors.
        prefix : string
            Key prefix to use for RediSearch index creation.
        distance_metric : optional string
            Distance metric to use for Vector Search. Defaults to 'COSINE'.
        """

        vector_field = VectorField(
            "vector",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": 300,
                "DISTANCE_METRIC": distance_metric,
                "INITIAL_CAP": number_of_vectors,
            },
        )

        await self._create(
            *fields,
            vector_field,
            index_name=index_name,
            redis_conn=redis_conn,
            prefix=prefix,
        )

    async def _create(self, *fields, index_name: str, redis_conn: Redis, prefix: str):
        """Create an index.

        Parameters
        ----------
        index_name : string
            The name of the index to be created.
        """

        await redis_conn.ft(index_name).create_index(
            fields=fields,
            definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
        )

    def process_tags(self, categories: list, years: list) -> str:
        """Helper function to process tags data.

        Parameters
        ----------
        categories : list
            List of categories.
        years : list
            List of years.

        Returns
        -------
        str
            RediSearch tag query string.
        """

        tag = "("
        if years:
            years_str = "|".join([self.escaper.escape(year) for year in years])
            tag += f"(@year:{{{years_str}}})"

        if categories:
            categories_str = "|".join([self.escaper.escape(cat) for cat in categories])
            if tag:
                tag += f" (@categories:{{{categories_str}}})"
            else:
                tag += f"(@categories:{{{categories_str}}})"

        tag += ")"

        # If no tags are selected, select all keys.
        if len(tag) < 3:
            tag = "*"

        return tag

    def vector_query(
        self,
        categories: list,
        years: list,
        search_type: str = "KNN",
        number_of_results: int = 20,
    ) -> Query:
        """Create a RediSearch query to perform hybrid vector and tag based searches.

        Parameters
        ----------
        categories : list
            List of categories.
        years : list
            List of years.
        search_type : optional str
            Style of search. Defaults to "KNN".
        number_of_results : optional int
            The number of results to fetch. Defaults to 20.

        Returns
        -------
        Query
            The RediSearch Query object.
        """
        # Parse tags to create query.
        tag_query = self.process_tags(categories, years)

        # Use the tag_query and feed the results from it into the KNN query.
        base_query = f"{tag_query}=>[{search_type} {number_of_results} @vector $vec_param AS vector_score]"

        return (
            Query(base_query)
            .sort_by("vector_score")
            .paging(0, number_of_results)
            .return_fields("paper_id", "paper_pk", "vector_score")
            .dialect(2)
        )

    def count_query(self, years: list, categories: list) -> Query:
        """Create a RediSearch query to count available documents.

        Parameters
        ----------
        categories : list
            List of categories.
        years : list
            List of years.

        Returns
        -------
        Query
            The RediSearch Query object.
        """

        # Parse tags to create query
        tag_query = self.process_tags(categories, years)

        return Query(f"{tag_query}").no_content().dialect(2)
