"""ListLabels on both facades (#860), as client-java's ``listLabels``.

The labels come back sorted, as the server sends them, with a count only when it was asked for: without it every
count is 0, which means "not counted".
"""

from __future__ import annotations

import asyncio
from typing import Any

import grpc
import pytest

from cyrock_db import CyrockDbClient, NotFoundException
from cyrock_db import _proto_converters as converters
from cyrock_db import _proto_requests as requests
from cyrock_db._proto import cyrock_db_graph_pb2 as graph_pb2
from cyrock_db.aio import AsyncCyrockDbClient
from cyrock_db.types import LabelStat, Vector

GRAPH = "graph-1"


def _client(server: Any) -> CyrockDbClient:
    return CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key")


def _seed(client: CyrockDbClient) -> None:
    client.add_node(GRAPH, ["Person"], {"embedding": Vector([1.0])}, None)
    client.add_node(GRAPH, ["Person", "Employee"], {"embedding": Vector([2.0])}, None)
    client.add_node(GRAPH, ["Company"], {"embedding": Vector([3.0])}, None)


def test_listLabels_includeCounts_isSentWithTheGraphId() -> None:
    counted   = requests.list_labels(GRAPH, include_counts=True)
    uncounted = requests.list_labels(GRAPH)

    assert counted.graph_id == GRAPH
    assert counted.include_counts is True
    assert uncounted.include_counts is False


def test_toLabelStats_response_mapsEveryLabelInOrder() -> None:
    response = graph_pb2.ListLabelsResponse(labels=[
        graph_pb2.LabelStatProto(label="A", count=3),
        graph_pb2.LabelStatProto(label="B"),
    ])

    assert converters.to_label_stats(response) == (LabelStat("A", 3), LabelStat("B", 0))


def test_toLabelStats_emptyResponse_returnsAnEmptyTuple() -> None:
    assert converters.to_label_stats(graph_pb2.ListLabelsResponse()) == ()


def test_listLabels_withCounts_returnsEveryLabelWithItsNodeCount(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        _seed(client)
        labels = client.list_labels(GRAPH, include_counts=True)

    assert labels == (LabelStat("Company", 1), LabelStat("Employee", 1), LabelStat("Person", 2))
    sent = server.graph.requests[-1]
    assert sent.graph_id == GRAPH
    assert sent.include_counts is True


def test_listLabels_defaultArguments_leavesCountsOut(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        _seed(client)
        labels = client.list_labels(GRAPH)

    assert labels == (LabelStat("Company", 0), LabelStat("Employee", 0), LabelStat("Person", 0))
    assert server.graph.requests[-1].include_counts is False


def test_listLabels_serverRefuses_raisesTheMappedException(token_server: Any) -> None:
    server = token_server(graph={"fail_with": grpc.StatusCode.NOT_FOUND})
    with _client(server) as client, pytest.raises(NotFoundException):
        client.list_labels("missing", include_counts=True)


def test_listLabels_asyncClient_returnsWhatTheBlockingClientDoes(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        _seed(client)
        blocking = client.list_labels(GRAPH, include_counts=True)

    async def run() -> tuple[tuple[LabelStat, ...], tuple[LabelStat, ...]]:
        async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key") as client:
            return await client.list_labels(GRAPH, include_counts=True), await client.list_labels(GRAPH)

    counted, uncounted = asyncio.run(run())

    assert counted == blocking
    assert uncounted == (LabelStat("Company", 0), LabelStat("Employee", 0), LabelStat("Person", 0))
    assert server.graph.requests[-1].include_counts is False
