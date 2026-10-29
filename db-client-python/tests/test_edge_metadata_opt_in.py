"""The edge-metadata opt-in for graph reads that return edges.

Neighbours, context windows, recall, reasoning chains and induced subgraphs leave the edges' metadata out
unless the request asks for it, matching the Java SDK: without it the server answers the edges from its
adjacency index and loads none of them (#712). These assert on the encoded request only: the server does
the leaving out.
"""

from __future__ import annotations

from cyrock_db import _proto_requests as requests
from cyrock_db.types import (
    GraphRecallRequest,
    GraphSearchExpansionRequest,
    NeighborsRequest,
    ReasoningChainRequest,
)


def test_neighbors_includeEdgeMetadata_isOffByDefaultAndEncodedWhenSet() -> None:
    assert requests.neighbors("g1", NeighborsRequest(1)).include_edge_metadata is False
    assert requests.neighbors(
        "g1", NeighborsRequest(1, include_edge_metadata=True),
    ).include_edge_metadata is True


def test_contextWindow_includeEdgeMetadata_isOffByDefaultAndEncodedWhenSet() -> None:
    assert requests.context_window(
        "g1", GraphSearchExpansionRequest("embedding", (1.0,), 5, 2),
    ).include_edge_metadata is False
    assert requests.context_window(
        "g1", GraphSearchExpansionRequest("embedding", (1.0,), 5, 2, include_edge_metadata=True),
    ).include_edge_metadata is True


def test_recall_includeEdgeMetadata_isOffByDefaultAndEncodedWhenSet() -> None:
    assert requests.recall(
        "g1", GraphRecallRequest("embedding", (1.0,), 5, 24),
    ).include_edge_metadata is False
    assert requests.recall(
        "g1", GraphRecallRequest("embedding", (1.0,), 5, 24, include_edge_metadata=True),
    ).include_edge_metadata is True


def test_reasoningChain_includeEdgeMetadata_isOffByDefaultAndEncodedWhenSet() -> None:
    assert requests.reasoning_chain("g1", ReasoningChainRequest(1, 2, 5)).include_edge_metadata is False
    assert requests.reasoning_chain(
        "g1", ReasoningChainRequest(1, 2, 5, include_edge_metadata=True),
    ).include_edge_metadata is True


def test_inducedSubgraph_includeEdgeMetadata_isOffByDefaultAndEncodedWhenSet() -> None:
    assert requests.induced_subgraph("g1", [1, 2]).include_edge_metadata is False
    built = requests.induced_subgraph("g1", [1, 2], include_vectors=True, include_edge_metadata=True)

    assert built.include_edge_metadata is True
    assert built.include_vectors is True
