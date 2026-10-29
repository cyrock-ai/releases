"""The bound and the vector opt-in for traversals and context windows.

A traversal returns at most ``limit`` nodes and, like a context window, leaves the nodes' vectors out unless
the request asks for them, matching the Java SDK. These assert on the encoded request only: the server does
the bounding and the leaving out.
"""

from __future__ import annotations

from cyrock_db import _proto_requests as requests
from cyrock_db.types import Direction, GraphSearchExpansionRequest, TraverseRequest


def test_traverse_defaults_noLimitAndNoVectors() -> None:
    built = requests.traverse("g1", TraverseRequest(1, 3))

    assert built.limit == 0
    assert built.include_vectors is False


def test_traverse_limitAndVectors_areEncodedWhenSet() -> None:
    built = requests.traverse(
        "g1", TraverseRequest(1, 3, Direction.OUTGOING, "KNOWS", limit=50, include_vectors=True),
    )

    assert built.limit == 50
    assert built.include_vectors is True
    assert built.edge_type == "KNOWS"


def test_contextWindow_includeVectors_isOffByDefaultAndEncodedWhenSet() -> None:
    assert requests.context_window(
        "g1", GraphSearchExpansionRequest("embedding", (1.0,), 5, 2),
    ).include_vectors is False
    assert requests.context_window(
        "g1", GraphSearchExpansionRequest("embedding", (1.0,), 5, 2, include_vectors=True),
    ).include_vectors is True
