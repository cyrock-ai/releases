"""Graph nodes, edges and the requests that address them.

Mirrors ``Node``, ``Edge``, ``AddEdgeRequest``, ``UpsertNodeRequest``, ``ListNodesRequest``,
``GraphWindow``, ``MergeResult`` and the ``LabelMatch`` enum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._enums import Direction
from ._filter_parameters import copy_filter_parameters
from ._values import VectorInput

__all__ = [
    "TraverseRequest",
    "SearchSimilarRequest",
    "ReasoningChainRequest",
    "NodeMatch",
    "NeighborsRequest",
    "GraphSearchExpansionRequest",
    "GraphRecallRequest",
    "AddEdgeRequest",
    "Edge",
    "GraphWindow",
    "LabelMatch",
    "ListNodesRequest",
    "MergeResult",
    "Node",
    "UpsertNodeRequest",
    "UpsertNodeResult",
]


class LabelMatch(Enum):
    """Whether a multi-label filter requires every label or any of them."""

    ALL = "ALL"
    ANY = "ANY"


@dataclass(frozen=True)
class Node:
    """A graph node: its labels, optional application key, vectors and metadata."""

    id:               int                             = 0
    labels:           tuple[str, ...]                 = ()
    external_key:     str | None                      = None
    vectors:          Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    metadata:         Mapping[str, Any]               = field(default_factory=dict)
    created_at:       int                             = 0
    #: Epoch millis of the last recall. A stored node on a server that populates this field always carries
    #: a real value - seeded from ``created_at`` on add, advanced by every recall - so 0 means the value is
    #: not known here: locally constructed, or a response from a server that does not send it. What
    #: ``recall`` scores recency from, so a node retrieved repeatedly keeps a recent value here.
    last_accessed_at: int                             = 0

    def __post_init__(self) -> None:
        # Duplicates are removed and order kept, matching Java's constructor. dict.fromkeys is the
        # order-preserving spelling of "distinct" in Python.
        object.__setattr__(self, "labels", tuple(dict.fromkeys(self.labels or ())))
        object.__setattr__(self, "vectors", {
            name: tuple(float(each) for each in values) for name, values in (self.vectors or {}).items()
        })
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class Edge:
    """A directed, weighted, typed edge between two nodes."""

    id:         int               = 0
    type:       str               = ""
    source_id:  int               = 0
    target_id:  int               = 0
    weight:     float             = 0.0
    metadata:   Mapping[str, Any] = field(default_factory=dict)
    created_at: int               = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class AddEdgeRequest:
    """An edge to create: its endpoints, type, weight and metadata."""

    source_id: int
    target_id: int
    type:      str
    weight:    float             = 1.0
    metadata:  Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class UpsertNodeRequest:
    """A node to insert, or to fully replace the node already holding an external key.

    With ``external_key`` set, the key decides: a new key inserts, an existing one replaces the node
    holding it. With no key, the store assigns an id and always creates. To replace a node addressed
    by its id, use :class:`UpdateNodeRequest` - that form cannot create, because node ids are assigned
    by the store.

    A replace is total, not a merge: labels, vectors and metadata are all overwritten, and a vector
    field the node held but this request omits is cleared.
    """

    labels:       Sequence[str]                                  = ()
    vectors:      Mapping[str, VectorInput | Sequence[float]]    = field(default_factory=dict)
    metadata:     Mapping[str, Any]                              = field(default_factory=dict)
    external_key: str | None                                     = None


@dataclass(frozen=True)
class UpdateNodeRequest:
    """A full replace of the node with ``node_id``, which must already exist.

    This is an update, never an insert: node ids are assigned by the store, so a caller cannot name an
    id that does not exist yet, and addressing an absent node raises
    :class:`~cyrock_db.CyrockDbClientException` rather than creating one.

    The replace is total, not a merge: labels, vectors and metadata are all overwritten, and a vector
    field the node held but this request omits is cleared. Pass the node's full intended state.

    :raises ValueError: if ``node_id`` is negative
    """

    node_id:  int
    labels:   Sequence[str]                               = ()
    vectors:  Mapping[str, VectorInput | Sequence[float]] = field(default_factory=dict)
    metadata: Mapping[str, Any]                           = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Zero is a real id - GigaMap numbers from zero, so node 0 is the first node of every graph.
        if self.node_id < 0:
            raise ValueError("node_id must not be negative")


@dataclass(frozen=True)
class UpsertNodeResult:
    """The outcome of an upsert by key: the node id, and whether it was created."""

    node_id: int
    created: bool


@dataclass(frozen=True)
class ListNodesRequest:
    """A page of nodes, optionally filtered by label and metadata.

    ``include_vectors`` is off by default, and deliberately so: collecting vectors is the most
    expensive part of building each node - at 384 dimensions with 200 neighbours, roughly 77k floats
    - and most readers never touch them. Asking makes the cost a deliberate choice rather than
    everybody's default.

    ``sample_seed`` turns the page into a seeded random sample of the scanned window. Without it a
    page is the first *n* nodes by id, and id order is insertion order, so a graph written one label
    at a time hands back a page whose nodes all share a label. Seeded rather than random, so a reload
    shows the same nodes instead of reshuffling under whoever is reading.

    ``filter`` is a CyQL metadata expression. Never format untrusted input into it: reference such values
    as ``$name`` and bind them in ``filter_parameters``, where each is compared as a literal.
    """

    offset:            int               = 0
    limit:             int               = 100
    labels:            Sequence[str]     = ()
    label_match:       LabelMatch        = LabelMatch.ALL
    exclude_labels:    Sequence[str]     = ()
    filter:            str | None        = None
    include_vectors:   bool              = False
    sample_seed:       int | None        = None
    filter_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "filter_parameters", copy_filter_parameters(self.filter_parameters))


@dataclass(frozen=True)
class GraphWindow:
    """A slice of a graph: nodes and the edges connecting them."""

    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))


@dataclass(frozen=True)
class MergeResult:
    """What a branch merge changed in the parent graph."""

    nodes_added:   int = 0
    nodes_removed: int = 0
    nodes_updated: int = 0
    edges_added:   int = 0
    edges_removed: int = 0
    edges_updated: int = 0


@dataclass(frozen=True)
class NodeMatch:
    """One similarity hit: the node, and the score it matched with."""

    score: float
    node:  Node


@dataclass(frozen=True)
class SearchSimilarRequest:
    """Approximate-nearest-neighbour search over node vectors, with optional label filtering.

    ``filter`` is a CyQL metadata expression. Never format untrusted input into it: reference such values
    as ``$name`` and bind them in ``filter_parameters``, where each is compared as a literal.
    """

    field_name:        str
    vector:            tuple[float, ...]
    max_results:       int
    labels:            Sequence[str]     = ()
    label_match:       LabelMatch        = LabelMatch.ALL
    exclude_labels:    Sequence[str]     = ()
    filter:            str | None        = None
    filter_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector",            tuple(float(each) for each in self.vector))
        object.__setattr__(self, "filter_parameters", copy_filter_parameters(self.filter_parameters))


@dataclass(frozen=True)
class NeighborsWindow:
    """The neighbours of a node as nodes and their edges, plus whether the limit dropped any.

    ``truncated`` is true when the server bounded the result to ``NeighborsRequest.limit`` and left
    neighbours out, so a caller can say "573 neighbours, showing 200". :class:`GraphWindow` carries
    no such flag, which is why :meth:`CyrockDbClient.neighbors_graph` returns this instead.
    """

    nodes:     tuple[Node, ...] = ()
    edges:     tuple[Edge, ...] = ()
    truncated: bool             = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))


@dataclass(frozen=True)
class NeighborsRequest:
    """The nodes one hop from a starting node, in a direction and optionally of one edge type.

    ``limit`` bounds the number of neighbours returned, not counting the queried node itself. Zero
    (the default) means no limit, and a hub node then returns every neighbour it has; a negative
    value is refused by the server. Whether the server dropped any is reported by :attr:`NeighborsWindow.truncated` from
    :meth:`CyrockDbClient.neighbors_graph`.

    ``include_edge_metadata`` asks for the edges' metadata and creation time, which are left out by default:
    without them the server answers the edges from its adjacency index and loads none of them, which on a hub
    is most of the read (#712). An edge read without metadata has an empty metadata map and a ``created_at``
    of 0.
    """

    node_id:               int
    direction:             Direction  = Direction.BOTH
    edge_type:             str | None = None
    diversity_seed:        int | None = None
    include_vectors:       bool       = False
    limit:                 int        = 0
    include_edge_metadata: bool       = False


@dataclass(frozen=True)
class TraverseRequest:
    """Breadth-first traversal from a starting node, to a maximum depth in hops.

    ``limit`` caps the returned nodes, the start node included; zero means no limit. With a limit the server stops
    walking once it has the nodes it returns. A negative ``limit`` or ``max_depth`` is refused by the server.
    ``include_vectors`` asks for the nodes' vectors, which are left out by default because they are the most
    expensive part of each node.
    """

    start_id:        int
    max_depth:       int
    direction:       Direction  = Direction.BOTH
    edge_type:       str | None = None
    limit:           int        = 0
    include_vectors: bool       = False


def _require_one_query(
    vector: tuple[float, ...],
    text:   str | None,
) -> None:
    """Refuses a request that carries neither or both of a query vector and query text, or blank text."""
    if text is None:
        if not vector:
            raise ValueError("either a non-empty vector or text is required")
        return
    if vector:
        raise ValueError("vector and text are mutually exclusive, set only one")
    if not text.strip():
        raise ValueError("text must not be blank")


@dataclass(frozen=True)
class GraphSearchExpansionRequest:
    """Vector search followed by graph expansion.

    ``depth`` is a **traversal hop count**. Contrast :class:`GraphRecallRequest`, whose
    ``recency_hours`` is a time window: the two used to share one field and one name, which is what
    issue #230 was about.

    The query is either ``vector`` or ``text``, which the server embeds with the field's embedding
    model; pass an empty ``vector`` with ``text``. ``include_vectors`` asks for the nodes' vectors, which are
    left out by default.

    ``include_edge_metadata`` asks for the edges' metadata and creation time, which are left out by default:
    without them the server answers the edges from its adjacency index and loads none of them, which on a hub
    is most of the read (#712). An edge read without metadata has an empty metadata map and a ``created_at``
    of 0.

    :raises ValueError: if neither or both of a non-empty ``vector`` and ``text`` are given, or ``text`` is
        blank
    """

    field_name:            str
    vector:                tuple[float, ...]
    max_results:           int
    depth:                 int
    diversity_seed:        int | None = None
    text:                  str | None = None
    include_vectors:       bool       = False
    include_edge_metadata: bool       = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(each) for each in self.vector))
        _require_one_query(self.vector, self.text)


@dataclass(frozen=True)
class GraphRecallRequest:
    """Memory recall: vector search re-weighted by recency, within a recency window.

    ``recency_hours`` is a **time window in hours**, not a traversal depth - recall drops memories
    older than it and re-weights the rest. It is required and must be positive: a zero-hour window is
    older than every memory and would filter out every result.

    The query is either ``vector`` or ``text``, which the server embeds with the field's embedding
    model; pass an empty ``vector`` with ``text``.

    ``include_edge_metadata`` asks for the edges' metadata and creation time, which are left out by default:
    without them the server answers the edges from its adjacency index and loads none of them, which on a hub
    is most of the read (#712). An edge read without metadata has an empty metadata map and a ``created_at``
    of 0.

    :raises ValueError: if ``recency_hours`` is not positive, neither or both of a non-empty
        ``vector`` and ``text`` are given, or ``text`` is blank
    """

    field_name:            str
    vector:                tuple[float, ...]
    max_results:           int
    recency_hours:         int
    diversity_seed:        int | None = None
    text:                  str | None = None
    include_edge_metadata: bool       = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(each) for each in self.vector))
        _require_one_query(self.vector, self.text)
        if self.recency_hours <= 0:
            raise ValueError(f"recency_hours must be positive, was {self.recency_hours}")


@dataclass(frozen=True)
class ReasoningChainRequest:
    """The shortest path between two nodes, up to a maximum depth.

    ``include_edge_metadata`` asks for the edges' metadata and creation time, which are left out by default:
    without them the server answers the edges from its adjacency index and loads none of them, which on a hub
    is most of the read (#712). An edge read without metadata has an empty metadata map and a ``created_at``
    of 0.
    """

    from_id:               int
    to_id:                 int
    max_depth:             int
    include_edge_metadata: bool = False
