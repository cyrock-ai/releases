"""The opt-in for search matches' vectors.

Collection searches leave each match's vectors out unless the request asks for them, matching the graph
reads and the Java SDK. These assert on the encoded request only: the server does the leaving out.
"""

from __future__ import annotations

from cyrock_db import _proto_requests as requests
from cyrock_db.types import (
    FusionStrategy,
    HybridSearchRequest,
    MultiVectorSearchRequest,
    SearchRequest,
    TextQuery,
    VectorQuery,
)


def test_search_includeVectors_isOffByDefault() -> None:
    built = requests.search("c1", SearchRequest("embedding", (1.0,), 5))

    assert built.include_vectors is False


def test_search_includeVectors_isEncodedWhenSet() -> None:
    built = requests.search("c1", SearchRequest("embedding", (1.0,), 5, include_vectors=True))

    assert built.include_vectors is True


def test_multiSearch_includeVectors_isOffByDefaultAndEncodedWhenSet() -> None:
    queries = (VectorQuery("embedding", (1.0,)),)

    assert requests.multi_search("c1", MultiVectorSearchRequest(queries, 5)).include_vectors is False
    assert requests.multi_search(
        "c1", MultiVectorSearchRequest(queries, 5, FusionStrategy.RRF, include_vectors=True),
    ).include_vectors is True


def test_hybridSearch_includeVectors_isOffByDefaultAndEncodedWhenSet() -> None:
    texts = (TextQuery("content", "graph"),)

    assert requests.hybrid_search("c1", HybridSearchRequest(text_queries=texts)).include_vectors is False
    assert requests.hybrid_search(
        "c1", HybridSearchRequest(text_queries=texts, include_vectors=True),
    ).include_vectors is True
