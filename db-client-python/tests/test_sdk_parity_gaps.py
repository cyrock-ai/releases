"""Fields the server honours on both transports that the SDK used to leave unsendable or unreadable.

The traverse truncation flag (#885), rerank on multi and hybrid search (#886), the search diversity seed (#887)
and the unrestricted statement switch (#894). The Java SDK carries the same fields; see ``ProtoRequestsSdkParityTest``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from cyrock_db import CyrockDbClient
from cyrock_db import _proto_requests as requests
from cyrock_db.aio import AsyncCyrockDbClient
from cyrock_db.types import (
    HybridSearchRequest,
    MultiVectorSearchRequest,
    QueryParameters,
    RerankOptions,
    SearchSimilarRequest,
    TextQuery,
    TraverseRequest,
    VectorQuery,
)

GRAPH  = "graph-1"
RERANK = RerankOptions(enabled=True, field="text", candidates=20, query="query")


def test_searchSimilar_diversitySeed_isEncodedWhenSet() -> None:
    built = requests.search_similar(GRAPH, SearchSimilarRequest("embedding", (1.0, 0.0), 5, diversity_seed=7))

    assert built.HasField("diversity_seed")
    assert built.diversity_seed == 7


def test_searchSimilar_noDiversitySeed_leavesItUnset() -> None:
    assert not requests.search_similar(GRAPH, SearchSimilarRequest("embedding", (1.0, 0.0), 5)).HasField(
        "diversity_seed"
    )


def test_multiSearch_rerank_isEncodedWhenEnabled() -> None:
    request = MultiVectorSearchRequest((VectorQuery("embedding", (1.0,), 1.0),), 5, rerank=RERANK)

    built = requests.multi_search("c1", request)

    assert built.HasField("rerank")
    assert built.rerank.field == "text"


def test_multiSearch_noRerank_leavesItUnset() -> None:
    built = requests.multi_search("c1", MultiVectorSearchRequest((VectorQuery("embedding", (1.0,), 1.0),), 5))

    assert not built.HasField("rerank")


def test_hybridSearch_rerank_isEncodedWhenEnabled() -> None:
    built = requests.hybrid_search(
        "c1", HybridSearchRequest(text_queries=(TextQuery("content", "machine learning", 1.0),), rerank=RERANK)
    )

    assert built.HasField("rerank")
    assert built.rerank.candidates == 20


def test_hybridSearch_noRerank_leavesItUnset() -> None:
    assert not requests.hybrid_search("c1", HybridSearchRequest()).HasField("rerank")


def test_executeStatement_unrestricted_isEncodedWhenSet() -> None:
    params = QueryParameters.builder().unrestricted(True).build()

    assert requests.execute_statement(GRAPH, "MATCH (n) DETACH DELETE n", params).unrestricted is True
    assert requests.execute_collection_statement("c1", "MATCH (d) DELETE d", params).unrestricted is True


def test_executeStatement_defaultIsRestricted() -> None:
    params = QueryParameters.builder().param("k", 1).build()

    assert requests.execute_statement(GRAPH, "MATCH (n) RETURN n", params).unrestricted is False
    assert requests.execute_collection_statement("c1", "MATCH (d) RETURN d", None).unrestricted is False


def test_traverseResult_limitedResult_exposesTruncated(token_server: Callable[..., Any]) -> None:
    server = token_server()
    with CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key") as client:
        start = client.add_node(GRAPH, ["N"], {}, {})
        client.add_node(GRAPH, ["N"], {}, {})
        limited   = client.traverse_result(GRAPH, TraverseRequest(start, max_depth=3, limit=1))
        roomy     = client.traverse_result(GRAPH, TraverseRequest(start, max_depth=3, limit=5))
        unlimited = client.traverse_result(GRAPH, TraverseRequest(start, max_depth=3))

    assert limited.truncated is True
    assert len(limited.nodes) == 1
    assert roomy.truncated is False
    assert unlimited.truncated is False
    assert len(unlimited.nodes) == 2


def test_traverseResult_async_exposesTruncated(token_server: Callable[..., Any]) -> None:
    server = token_server()

    async def run() -> bool:
        async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key") as client:
            start = await client.add_node(GRAPH, ["N"], {}, {})
            await client.add_node(GRAPH, ["N"], {}, {})
            result = await client.traverse_result(GRAPH, TraverseRequest(start, max_depth=3, limit=1))
            return result.truncated

    assert asyncio.run(run()) is True


def test_executeProjectStatement_vectorParameter_isRefused() -> None:
    params = QueryParameters.builder().vector("v", (1.0, 0.0)).build()

    with pytest.raises(ValueError, match="no vector parameters"):
        requests.execute_project_statement("p1", "SHOW GRAPHS", params)
