"""Both transaction surfaces: what each operation encodes to, and what comes back.

Both are all-or-nothing: a failing operation fails the whole call, so a returned result only ever
describes a committed transaction. Its per-operation results carry the ids the operations generated,
and ``None`` for an operation that generated nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cyrock_db import CyrockDbClient
from cyrock_db._proto_requests import execute_document_transaction, execute_transaction
from cyrock_db.aio import AsyncCyrockDbClient
from cyrock_db.types import (
    AddEdgeOp,
    AddNodeOp,
    EndpointKey,
    EndpointNodeId,
    EndpointOperationIndex,
    OperationResult,
    RemoveDocumentOp,
    RemoveEdgeOp,
    RemoveNodeOp,
    TransactionResult,
    UpdateEdgeMetadataOp,
    UpdateNodeMetadataOp,
    UpdateNodeOp,
    UpdateVectorOp,
    UpdateWeightOp,
    UpsertDocumentOp,
    UpsertKeyOp,
    UpsertNodeOp,
    Vector,
)

COLLECTION = "collection-1"
GRAPH      = "graph-1"


def _client(server: Any) -> CyrockDbClient:
    return CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key")


# ─── Encoding ──────────────────────────────────────────────────────────────


def test_documentOperations_eachEncodeToTheirOwnArm() -> None:
    request = execute_document_transaction(COLLECTION, [
        UpsertDocumentOp(id=-1, vectors={"embedding": Vector([1.0])}, metadata={"n": 1}),
        UpsertKeyOp(external_key="k", vectors={"embedding": Vector([2.0])}),
        RemoveDocumentOp(document_id=7),
    ])

    assert [each.WhichOneof("operation") for each in request.operations] == [
        "upsert", "upsert_by_key", "remove",
    ]
    assert request.operations[0].upsert.id == -1
    assert request.operations[1].upsert_by_key.external_key == "k"
    assert request.operations[2].remove.document_id == 7


def test_graphOperations_eachEncodeToTheirOwnArm() -> None:
    request = execute_transaction(GRAPH, [
        AddNodeOp(labels=["A"], vectors={"embedding": Vector([1.0])}, metadata={"n": 1}),
        AddEdgeOp(1, 2, "T", 0.5, {"why": "x"}),
        RemoveNodeOp.by_id(3),
        RemoveNodeOp.by_key("k"),
        RemoveEdgeOp(4),
        UpdateVectorOp(5, "embedding", [1.0, 2.0]),
        UpdateWeightOp(6, 0.25),
        UpdateNodeMetadataOp(7, {"v": 2}),
        UpdateEdgeMetadataOp(8, {"since": 2099}),
        UpsertNodeOp.create(labels=["B"]),
        UpdateNodeOp(9, ["C"]),
    ])

    assert [each.WhichOneof("operation") for each in request.operations] == [
        "add_node", "add_edge", "remove_node", "remove_node_by_key", "remove_edge",
        "update_vector", "update_weight", "update_node_metadata", "update_edge_metadata", "upsert_node",
        "update_node",
    ]


def test_addEdgeOp_betweenOperations_encodesOpIndicesOnly() -> None:
    request = execute_transaction(GRAPH, [
        AddNodeOp(labels=["Person"]),
        AddNodeOp(labels=["Person"]),
        AddEdgeOp(EndpointOperationIndex(0), EndpointOperationIndex(1), "KNOWS", 0.9, {"since": 2020}),
    ])

    edge = request.operations[2].add_edge
    assert edge.HasField("source_op_index") and edge.source_op_index == 0
    assert edge.HasField("target_op_index") and edge.target_op_index == 1
    assert not edge.HasField("source_id")
    assert not edge.HasField("target_id")
    assert edge.type == "KNOWS"
    assert edge.weight == pytest.approx(0.9)


def test_addEdgeOp_toExistingNode_mixesIdAndOpIndex() -> None:
    request = execute_transaction(GRAPH, [
        AddNodeOp(labels=["Doc"]),
        AddEdgeOp(EndpointNodeId(0), EndpointOperationIndex(0), "AUTHORED"),
    ])

    edge = request.operations[1].add_edge
    assert edge.HasField("source_id") and edge.source_id == 0
    assert not edge.HasField("source_op_index")
    assert edge.HasField("target_op_index") and edge.target_op_index == 0
    assert not edge.HasField("target_id")
    assert edge.weight == pytest.approx(1.0)


def test_addEdgeOp_plainIntAndEndpointNodeId_encodeTheSame() -> None:
    request = execute_transaction(GRAPH, [AddEdgeOp(1, EndpointNodeId(2), "T")])

    edge = request.operations[0].add_edge
    assert edge.WhichOneof("source") == "source_id" and edge.source_id == 1
    assert edge.WhichOneof("target") == "target_id" and edge.target_id == 2


def test_addEdgeOp_byKey_encodesKeysOnly() -> None:
    request = execute_transaction(GRAPH, [
        AddNodeOp(labels=["Post"]),
        AddEdgeOp(EndpointOperationIndex(0), EndpointKey("person:1"), "HAS_CREATOR"),
        AddEdgeOp(EndpointKey("forum:2"), EndpointOperationIndex(0), "CONTAINER_OF"),
    ])

    creator = request.operations[1].add_edge
    assert creator.WhichOneof("source") == "source_op_index"
    assert creator.WhichOneof("target") == "target_key" and creator.target_key == "person:1"
    container = request.operations[2].add_edge
    assert container.WhichOneof("source") == "source_key" and container.source_key == "forum:2"


@pytest.mark.parametrize("external_key", ["", "  ", None, 7])
def test_endpointKey_blankOrNotAString_raisesValueError(external_key: Any) -> None:
    with pytest.raises(ValueError, match="must be a non-blank string"):
        EndpointKey(external_key)


def test_endpointOperationIndex_negative_raisesValueError() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        EndpointOperationIndex(-1)


def test_removeNodeOp_takesExactlyOneAddress() -> None:
    """Java offers byId/byKey factories over one record with two nullable fields; both cannot be set."""
    with pytest.raises(ValueError, match="exactly one"):
        RemoveNodeOp(node_id=1, external_key="k")
    with pytest.raises(ValueError, match="exactly one"):
        RemoveNodeOp()


@pytest.mark.parametrize(
    ("operation", "expects_external_key"),
    [
        (UpsertNodeOp.by_key("k", ["A"]), True),
        (UpsertNodeOp.create(["A"]),      False),
    ],
)
def test_upsertNodeOp_addressing(operation: UpsertNodeOp, expects_external_key: bool) -> None:
    """`create` sets no key, so the store assigns an id and always creates."""
    encoded = execute_transaction(GRAPH, [operation]).operations[0].upsert_node
    assert encoded.HasField("external_key") is expects_external_key


def test_updateNodeOp_carriesItsNodeId() -> None:
    """The by-id form is its own operation: it replaces an existing node and can never create one."""
    encoded = execute_transaction(GRAPH, [UpdateNodeOp(9, ["A"])]).operations[0].update_node
    assert encoded.HasField("node_id") is True
    assert encoded.node_id == 9


def test_updateNodeOp_nodeIdZero_isARealId() -> None:
    """GigaMap numbers from zero, so node 0 must survive the round trip as a set field, not an absent one."""
    encoded = execute_transaction(GRAPH, [UpdateNodeOp(0, ["A"])]).operations[0].update_node
    assert encoded.HasField("node_id") is True
    assert encoded.node_id == 0


def test_updateNodeOp_negativeNodeId_isRefused() -> None:
    """There is no auto-assign sentinel here: the node must already exist."""
    with pytest.raises(ValueError, match="must not be negative"):
        UpdateNodeOp(-1, ["A"])


def test_anUnknownOperation_isRefusedRatherThanDropped() -> None:
    """The failure mode this guards is silence: a dropped operation in an all-or-nothing surface."""
    class Invented:
        pass

    with pytest.raises(TypeError, match="Unsupported graph operation"):
        execute_transaction(GRAPH, [Invented()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Unsupported collection operation"):
        execute_document_transaction(COLLECTION, [Invented()])  # type: ignore[list-item]


# ─── Round trip ────────────────────────────────────────────────────────────


def test_documentTransaction_returnsOneResultPerOperation(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        result = client.execute_document_transaction(COLLECTION, [
            UpsertDocumentOp(id=-1, vectors={"embedding": Vector([1.0])}),
            RemoveDocumentOp(document_id=1),
        ])

    assert isinstance(result, TransactionResult)
    assert [each.index for each in result.results] == [0, 1]
    assert [each.generated_id for each in result.results] == [500, None]


def test_graphTransaction_returnsOneResultPerOperation(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        result = client.execute_transaction(GRAPH, [AddNodeOp(labels=["A"]), RemoveEdgeOp(1)])

    assert [each.generated_id for each in result.results] == [1000, None]


def test_anEmptyTransaction_isAcceptedAndReturnsNothing(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        assert client.execute_transaction(GRAPH, []).results == ()


def test_operationResult_noGeneratedId_isNoneRatherThanZero() -> None:
    assert OperationResult(index=0).generated_id is None
    assert OperationResult(index=0, generated_id=0).generated_id == 0


def test_bothFacades_agreeOnTransactionResults(token_server: Any) -> None:
    server     = token_server()
    operations = [AddNodeOp(labels=["A"]), AddEdgeOp(1, 2, "T")]
    with _client(server) as client:
        blocking = client.execute_transaction(GRAPH, operations)

    async def awaited() -> TransactionResult:
        async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key") as client:
            return await client.execute_transaction(GRAPH, operations)

    assert blocking == asyncio.run(awaited())
