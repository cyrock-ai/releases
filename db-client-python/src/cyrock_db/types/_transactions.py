"""Transaction operations and their results.

Both transaction surfaces are atomic: every operation commits or none does, applied through an undo
log and one atomic commit. A failing operation fails the whole call with an exception, so a result
only ever describes a committed transaction; its per-operation entries carry the ids the operations
generated, and ``None`` for an operation that generated nothing.

Java models the operations as sealed interfaces with nested records
(``CyrockDbClient.GraphOperation``, ``CyrockDbClient.CollectionOperation``). Python has no sealed
types; these are frozen dataclasses in a shared union, which gives the same closed set for type
checking and reads the same at a call site.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._values import VectorInput

__all__ = [
    "AddEdgeOp",
    "AddNodeOp",
    "CollectionOperation",
    "EdgeEndpoint",
    "EndpointKey",
    "EndpointNodeId",
    "EndpointOperationIndex",
    "GraphOperation",
    "OperationResult",
    "RemoveDocumentOp",
    "RemoveEdgeOp",
    "RemoveNodeOp",
    "TransactionResult",
    "UpdateEdgeMetadataOp",
    "UpdateNodeMetadataOp",
    "UpdateVectorOp",
    "UpdateWeightOp",
    "UpsertDocumentOp",
    "UpsertKeyOp",
    "UpsertNodeOp",
]

Vectors = Mapping[str, VectorInput | Sequence[float]]


# ─── Collection operations ──────────────────────────────────────────────────


@dataclass(frozen=True)
class UpsertDocumentOp:
    """Insert a document, or replace the one with this id.

    A negative id asks the store to assign one. A non-negative id is update-only: it must name an existing
    document, or the whole transaction fails as not found.
    """

    id:       int
    vectors:  Vectors           = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpsertKeyOp:
    """Insert or replace a document by the caller's stable key."""

    external_key: str
    vectors:      Vectors           = field(default_factory=dict)
    metadata:     Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoveDocumentOp:
    """Remove a document by id."""

    document_id: int


CollectionOperation = UpsertDocumentOp | UpsertKeyOp | RemoveDocumentOp


# ─── Graph operations ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class AddNodeOp:
    """Add a labelled node."""

    labels:   Sequence[str]     = ()
    vectors:  Vectors           = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointNodeId:
    """An :class:`AddEdgeOp` endpoint naming an existing node by id.

    Java's ``GraphOperation.EdgeEndpoint.NodeId``, built there by ``EdgeEndpoint.node``.
    """

    node_id: int


@dataclass(frozen=True)
class EndpointOperationIndex:
    """An :class:`AddEdgeOp` endpoint naming the node an earlier operation in the same transaction creates.

    ``index`` is the zero-based position of an :class:`AddNodeOp` or :class:`UpsertNodeOp` that precedes the
    edge in the operation list. Java's ``GraphOperation.EdgeEndpoint.OperationIndex``, built there by
    ``EdgeEndpoint.operation``.

    :raises ValueError: if ``index`` is negative
    """

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must not be negative: {self.index}")


@dataclass(frozen=True)
class EndpointKey:
    """An :class:`AddEdgeOp` endpoint naming the node that holds an external key.

    The key is resolved when the edge's operation runs, so it also finds a node an earlier
    :class:`UpsertNodeOp` in the same transaction created; a key no node holds fails the whole transaction
    as not found. Java's ``GraphOperation.EdgeEndpoint.ExternalKey``, built there by ``EdgeEndpoint.key``.

    :raises ValueError: if ``external_key`` is not a string or is blank
    """

    external_key: str

    def __post_init__(self) -> None:
        # Checked at runtime as well as by the annotation: a non-string would otherwise surface as an
        # AttributeError here or a protobuf TypeError when the transaction is encoded.
        if not isinstance(self.external_key, str) or not self.external_key.strip():
            raise ValueError("external_key must be a non-blank string")


EdgeEndpoint = EndpointNodeId | EndpointOperationIndex | EndpointKey


@dataclass(frozen=True)
class AddEdgeOp:
    """Add a weighted, typed edge.

    Each endpoint is an existing node, :class:`EndpointNodeId` or simply its ``int`` id, the node an earlier
    operation in the same transaction creates, :class:`EndpointOperationIndex`, or the node holding an external
    key, :class:`EndpointKey`. A node's id is assigned only when the transaction runs, so the operation form is
    what lets one transaction add a node together with its edges; the key form connects nodes known by their own
    keys without reading their ids first. A referenced index must belong to an earlier :class:`AddNodeOp` or
    :class:`UpsertNodeOp`, or the whole transaction fails as an invalid argument; a key no node holds fails it as
    not found::

        client.execute_transaction(graph_id, [
            AddNodeOp(labels=["Person"], metadata={"name": "Ada"}),                     # 0
            AddNodeOp(labels=["Person"], metadata={"name": "Bob"}),                     # 1
            AddEdgeOp(EndpointOperationIndex(0), EndpointOperationIndex(1), "KNOWS", 0.9),
            AddEdgeOp(EndpointOperationIndex(0), existing_id, "WORKS_AT"),
            AddEdgeOp(EndpointOperationIndex(1), EndpointKey("city:berlin"), "LIVES_IN"),
        ])

    Java spells the ``int`` shorthand as ``AddEdge``'s id-pair constructor.
    """

    source:   EdgeEndpoint | int
    target:   EdgeEndpoint | int
    type:     str
    weight:   float             = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoveNodeOp:
    """Remove a node, by id or by external key.

    Java offers ``RemoveNode.byId`` and ``RemoveNode.byKey`` factories over one record with two
    nullable fields; the same shape holds here, and exactly one of the two should be set.
    """

    node_id:      int | None = None
    external_key: str | None = None

    def __post_init__(self) -> None:
        if (self.node_id is None) == (self.external_key is None):
            raise ValueError("RemoveNodeOp takes exactly one of node_id or external_key")

    @staticmethod
    def by_id(node_id: int) -> RemoveNodeOp:
        return RemoveNodeOp(node_id=node_id)

    @staticmethod
    def by_key(external_key: str) -> RemoveNodeOp:
        return RemoveNodeOp(external_key=external_key)


@dataclass(frozen=True)
class RemoveEdgeOp:
    """Remove an edge by id."""

    edge_id: int


@dataclass(frozen=True)
class UpdateVectorOp:
    """Replace one of a node's vectors."""

    node_id:    int
    field_name: str
    vector:     Sequence[float]


@dataclass(frozen=True)
class UpdateWeightOp:
    """Replace an edge's weight."""

    edge_id: int
    weight:  float


@dataclass(frozen=True)
class UpdateNodeMetadataOp:
    """Replace a node's metadata."""

    node_id:  int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateEdgeMetadataOp:
    """Replace an edge's metadata."""

    edge_id:  int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpsertNodeOp:
    """Insert a node, or fully replace the node already holding ``external_key``.

    With no key set the store assigns an id and always creates, which is what Java's
    ``UpsertNode.create`` spells. To replace a node addressed by its id, use :class:`UpdateNodeOp` -
    that form cannot create, because node ids are assigned by the store.
    """

    external_key: str | None        = None
    labels:       Sequence[str]     = ()
    vectors:      Vectors           = field(default_factory=dict)
    metadata:     Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def by_key(external_key: str, labels: Sequence[str] = (), vectors: Vectors | None = None,
               metadata: Mapping[str, Any] | None = None) -> UpsertNodeOp:
        return UpsertNodeOp(external_key=external_key, labels=labels, vectors=vectors or {},
                            metadata=metadata or {})

    @staticmethod
    def create(labels: Sequence[str] = (), vectors: Vectors | None = None,
               metadata: Mapping[str, Any] | None = None) -> UpsertNodeOp:
        return UpsertNodeOp(labels=labels, vectors=vectors or {}, metadata=metadata or {})


@dataclass(frozen=True)
class UpdateNodeOp:
    """Fully replace the node with ``node_id``, which must already exist.

    Update-only: node ids are assigned by the store, so a missing id fails the transaction rather than
    creating a node. Labels, vectors and metadata are all overwritten, and a vector field the node held
    but this operation omits is cleared.

    :raises ValueError: if ``node_id`` is negative
    """

    node_id:  int
    labels:   Sequence[str]     = ()
    vectors:  Vectors           = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Zero is a real id - GigaMap numbers from zero, so node 0 is the first node of every graph.
        if self.node_id < 0:
            raise ValueError("node_id must not be negative")


GraphOperation = (
    AddNodeOp | AddEdgeOp | RemoveNodeOp | RemoveEdgeOp
    | UpdateVectorOp | UpdateWeightOp | UpdateNodeMetadataOp | UpdateEdgeMetadataOp | UpsertNodeOp
    | UpdateNodeOp
)


# ─── Results ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OperationResult:
    """One operation's outcome in a committed transaction: its position and any id it generated.

    A transaction is atomic: a failing operation fails the whole call with an exception, so every
    operation in a returned result succeeded. ``generated_id`` is the id of the node, edge or document
    the operation created or upserted, and ``None`` when it produced none, as for a remove or an
    update - ``None`` rather than ``0``, since ``0`` is the id of the first entity.
    """

    index:        int
    generated_id: int | None = None


@dataclass(frozen=True)
class TransactionResult:
    """Per-operation results of a committed transaction, in the order the operations were submitted."""

    results: tuple[OperationResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
