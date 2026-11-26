"""Proto response -> domain type. Pure, no I/O.

One of the two layers both client facades share, matching ``client-java``'s ``ProtoConverters``.
Keeping it free of transport means the synchronous and asynchronous clients differ only in how they
issue the call, never in what a response means.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, TypeVar

from ._values_codec import from_struct
from .exceptions import CyrockDbClientException
from .types import (
    ApiKeyInfo,
    ApiKeyStatus,
    Cardinality,
    ChangeEvent,
    ChangeOp,
    ChangeSource,
    CollectionDefinition,
    CommunityAssignments,
    CommunityHierarchy,
    CommunityMemberSet,
    CommunityRef,
    CommunityResult,
    CommunityStatus,
    CommunitySummaries,
    CommunitySummary,
    CreatedApiKey,
    DecayResult,
    Document,
    Edge,
    EdgeInfo,
    EntityKind,
    GlobalSearchResult,
    GraphDefinition,
    GraphWindow,
    LabelStat,
    Match,
    MergeResult,
    MetadataFieldDefinition,
    MetadataFieldType,
    NeighborsWindow,
    NlMatch,
    NlSearchResult,
    Node,
    NodeMatch,
    ObserveResult,
    OperationResult,
    Page,
    ProjectInfo,
    RefreshCommunitiesResult,
    SchemaChange,
    SimilarityFunction,
    StatementClass,
    StatementResult,
    StatementType,
    TransactionResult,
    TraverseResult,
    VectorCompression,
    WriteSummary,
)

# The message a collection statement's summary is, see _to_statement_summary.
_COLLECTION_SUMMARY = "CollectionWriteSummaryProto"

__all__ = [
    "to_collection_definition",
    "to_document",
    "to_documents",
    "to_edge",
    "to_graph_window",
    "to_neighbors_window",
    "to_matches",
    "to_node",
    "to_node_matches",
    "to_statement_result",
    "to_transaction_result",
    "to_nodes",
    "to_label_stats",
    "to_collection_definitions",
    "to_graph_definition",
    "to_graph_definitions",
    "to_statement_class",
    "to_statement_type",
    "to_api_key_info",
    "to_api_key_infos",
    "to_created_api_key",
    "to_project_info",
    "to_project_infos",
]


def to_statement_class(name: str) -> StatementClass:
    """Parses the wire's ``statement_class`` string.

    ``statement_class`` and ``statement_type`` are declared ``string`` in the proto, not enums, and
    the Java converter parses them with ``StatementClass.valueOf`` - which throws on a name it does
    not know. This matches that strictness deliberately: a name we cannot parse means the server is
    newer than this client, and failing loudly beats quietly reporting the wrong statement class to
    a caller routing on it.

    :raises ValueError: if the name is not one this client knows
    """
    try:
        return StatementClass[name]
    except KeyError as unknown:
        raise ValueError(
            f"Unknown statement class '{name}'. The server is likely newer than this client; "
            f"upgrade cyrock-db."
        ) from unknown


def to_statement_type(name: str) -> StatementType:
    """Parses the wire's ``statement_type`` string. See :func:`to_statement_class`.

    :raises ValueError: if the name is not one this client knows
    """
    try:
        return StatementType[name]
    except KeyError as unknown:
        raise ValueError(
            f"Unknown statement type '{name}'. The server is likely newer than this client; "
            f"upgrade cyrock-db."
        ) from unknown


def _to_vector_field(proto: Any) -> Any:
    from .types._definitions import VectorFieldDefinition

    return VectorFieldDefinition(
        name                = proto.name,
        dimension           = proto.dimension,
        similarity_function = SimilarityFunction[proto.similarity_function]
                              if proto.similarity_function else None,  # type: ignore[arg-type]
        max_degree          = proto.max_degree,
        beam_width          = proto.beam_width,
        neighbor_overflow   = proto.neighbor_overflow,
        alpha               = proto.alpha,
        embedding_model     = proto.embedding_model or None,
        eventual_indexing   = proto.eventual_indexing,
        compression         = VectorCompression[proto.compression]
                              if proto.compression else None,  # type: ignore[arg-type]
    )


def _to_metadata_field(proto: Any) -> MetadataFieldDefinition:
    return MetadataFieldDefinition(
        name        = proto.name,
        type        = MetadataFieldType[proto.type],
        # A definition that came off the wire always names one; HIGH is the read-side fallback for a
        # server that somehow sent none, chosen because it is what an indexed field wants and because
        # refusing to decode a stored definition would be worse than assuming.
        cardinality = Cardinality[proto.cardinality] if proto.cardinality else Cardinality.HIGH,
        fulltext    = proto.fulltext,
        unique      = proto.unique,
    )


def to_collection_definition(proto: Any) -> CollectionDefinition:
    return CollectionDefinition(
        id              = proto.id or None,
        name            = proto.name or None,
        vector_fields   = tuple(_to_vector_field(each)   for each in proto.vector_fields),
        metadata_fields = tuple(_to_metadata_field(each) for each in proto.metadata_fields),
    )


def to_collection_definitions(proto: Any) -> tuple[CollectionDefinition, ...]:
    return tuple(to_collection_definition(each) for each in proto.collections)


def to_collection_definition_page(proto: Any) -> Page[CollectionDefinition]:
    """One page of a project's collections and the token of the next page, empty on the last."""
    return Page(items=to_collection_definitions(proto), next_page_token=proto.next_page_token)


def to_graph_definition(proto: Any) -> GraphDefinition:
    return GraphDefinition(
        id                       = proto.id or None,
        name                     = proto.name or None,
        node_vector_fields       = tuple(_to_vector_field(each)   for each in proto.node_vector_fields),
        node_metadata_fields     = tuple(_to_metadata_field(each) for each in proto.node_metadata_fields),
        edge_metadata_fields     = tuple(_to_metadata_field(each) for each in proto.edge_metadata_fields),
        enable_temporal_tracking = proto.enable_temporal_tracking,
        auto_link_threshold      = proto.auto_link_threshold,
        parent_graph_id          = proto.parent_graph_id or None,
        forked_at                = proto.forked_at,
        branch_name              = proto.branch_name or None,
    )


def to_graph_definitions(proto: Any) -> tuple[GraphDefinition, ...]:
    return tuple(to_graph_definition(each) for each in proto.graphs)


_DefinitionT = TypeVar("_DefinitionT", CollectionDefinition, GraphDefinition)


def sorted_by_id(definitions: Iterable[_DefinitionT]) -> tuple[_DefinitionT, ...]:
    """Sorts a whole listing by id, in the order the server pages in.

    The server compares ids as Java strings, by UTF-16 code unit, which differs from Python's code point order for a
    character outside the basic plane; comparing the UTF-16 big-endian encodings gives Java's order. A paging server's
    pages already are in it; the one answer of a server that predates paging is in its hash map's order.
    """
    return tuple(sorted(definitions, key=lambda definition: (definition.id or "").encode("utf-16-be")))


def to_graph_definition_page(proto: Any) -> Page[GraphDefinition]:
    """One page of a project's graphs and branches and the token of the next page, empty on the last."""
    return Page(items=to_graph_definitions(proto), next_page_token=proto.next_page_token)


def to_document(proto: Any) -> Document:
    """Decodes a ``DocumentProto``.

    Only the ``values`` half of each vector field is read: ``text`` is an *input*, asking the server
    to embed, and a stored document carries the embedding rather than the phrase that produced it.
    """
    return Document(
        id           = proto.id,
        vectors      = {
            name: tuple(field.values) for name, field in proto.vectors.items()
        },
        metadata     = from_struct(proto.metadata) if proto.HasField("metadata") else {},
        external_key = proto.external_key if proto.HasField("external_key") else None,
    )


def to_documents(proto: Any) -> tuple[Document, ...]:
    return tuple(to_document(each) for each in proto.documents)


def to_matches(proto: Any) -> tuple[Match, ...]:
    """Decodes a list of search hits."""
    return tuple(Match(score=each.score, document=to_document(each.document)) for each in proto.matches)


def to_node(proto: Any) -> Node:
    """Decodes a ``NodeProto``.

    ``vectors`` is empty when the read did not ask for them - which is the default on the listing and
    keyed-read paths - so an empty mapping means "not requested" as often as it means "none stored".
    """
    return Node(
        id           = proto.id,
        labels       = tuple(proto.labels),
        external_key = proto.external_key if proto.HasField("external_key") else None,
        vectors      = {name: tuple(field.values) for name, field in proto.vectors.items()},
        metadata     = from_struct(proto.metadata) if proto.HasField("metadata") else {},
        created_at   = proto.created_at,
        last_accessed_at = proto.last_accessed_at,
    )


def to_nodes(proto: Any) -> tuple[Node, ...]:
    return tuple(to_node(each) for each in proto.nodes)


def to_label_stats(proto: Any) -> tuple[LabelStat, ...]:
    """The labels of a ``ListLabelsResponse``, in the server's order; a count of 0 when counts were not asked for."""
    return tuple(LabelStat(label=each.label, count=each.count) for each in proto.labels)


def to_edge(proto: Any) -> Edge:
    return Edge(
        id         = proto.id,
        type       = proto.type,
        source_id  = proto.source_id,
        target_id  = proto.target_id,
        weight     = proto.weight,
        metadata   = from_struct(proto.metadata) if proto.HasField("metadata") else {},
        created_at = proto.created_at,
    )


def to_graph_window(proto: Any) -> GraphWindow:
    return GraphWindow(
        nodes=tuple(to_node(each) for each in proto.nodes),
        edges=tuple(to_edge(each) for each in proto.edges),
    )


def to_neighbors_window(proto: Any) -> NeighborsWindow:
    return NeighborsWindow(
        nodes=tuple(to_node(each) for each in proto.nodes),
        edges=tuple(to_edge(each) for each in proto.edges),
        truncated=proto.truncated,
    )


def to_traverse_result(proto: Any) -> TraverseResult:
    return TraverseResult(nodes=tuple(to_node(each) for each in proto.nodes), truncated=proto.truncated)


def to_node_matches(proto: Any) -> tuple[NodeMatch, ...]:
    """Decodes a list of similarity hits."""
    return tuple(NodeMatch(score=each.score, node=to_node(each.node)) for each in proto.matches)


def to_transaction_result(proto: Any) -> TransactionResult:
    """Decodes per-operation transaction results.

    An unset ``generated_id`` becomes ``None``: the operation produced no id, which a plain ``0``
    could not say, since ``0`` is the id of the first entity.
    """
    return TransactionResult(
        results=tuple(
            OperationResult(
                index        = each.index,
                generated_id = each.generated_id if each.HasField("generated_id") else None,
            )
            for each in proto.results
        )
    )


def _to_write_summary(proto: Any) -> WriteSummary:
    return WriteSummary(
        nodes_created    = proto.nodes_created,
        nodes_deleted    = proto.nodes_deleted,
        edges_created    = proto.edges_created,
        edges_deleted    = proto.edges_deleted,
        properties_set   = proto.properties_set,
        created_node_ids = tuple(proto.created_node_ids),
        created_edge_ids = tuple(proto.created_edge_ids),
    )


def _to_collection_write_summary(proto: Any) -> WriteSummary:
    """Decodes the summary of a collection write, which counts documents and has no edges.

    A collection answers with ``CollectionWriteSummaryProto``, whose fields are not the graph's. The
    result is the same :class:`WriteSummary` rather than a second type, as in client-java: the document
    counts fill the node counts, and the edge counts are zero.
    """
    return WriteSummary(
        nodes_created    = proto.documents_created,
        nodes_deleted    = proto.documents_deleted,
        edges_created    = 0,
        edges_deleted    = 0,
        properties_set   = proto.properties_set,
        created_node_ids = tuple(proto.created_document_ids),
        created_edge_ids = (),
    )


def _to_statement_summary(proto: Any) -> WriteSummary:
    """Decodes the summary of a statement response, which is a collection's or a graph's.

    Told apart by the message the response carries, not by probing for a field name: the two share no
    field but ``properties_set``, and a summary read through the wrong decoder raises on the first
    attribute the other one has (#893).
    """
    if proto.DESCRIPTOR.name == _COLLECTION_SUMMARY:
        return _to_collection_write_summary(proto)
    return _to_write_summary(proto)


def to_statement_result(proto: Any) -> StatementResult:
    """Decodes any of the three statement responses.

    They are deliberately separate messages - a collection has documents, a graph has nodes and
    edges, a project statement is DDL - but share one row encoding and these five field names, so one
    converter reads all three, as the Java one does.

    ``statement_class`` and ``statement_type`` are strings on the wire and are parsed strictly: an
    unrecognised name means the server is newer than this client, and failing loudly beats reporting
    the wrong class to a caller routing on it.
    """
    # hasattr first, then HasField: HasField raises on a message that has no such field, and
    # ProjectStatementResponse genuinely has no `summary` - a project statement is DDL and writes no
    # rows. The order matters, and the wrong one only fails on that one response type.
    summary = (
        _to_statement_summary(proto.summary)
        if hasattr(proto, "summary") and proto.HasField("summary")
        else None
    )
    definition = (
        to_graph_definition(proto.definition)
        if hasattr(proto, "definition") and proto.HasField("definition")
        else None
    )
    return StatementResult(
        statement_class = to_statement_class(proto.statement_class),
        statement_type  = to_statement_type(proto.statement_type),
        columns         = tuple(proto.columns),
        rows            = tuple(from_struct(row.values) for row in proto.rows),
        summary         = summary,
        definition      = definition,
    )


# ─── Branching ──────────────────────────────────────────────────────────────


def to_merge_result(proto: Any) -> MergeResult:
    return MergeResult(
        nodes_added   = proto.nodes_added,
        nodes_removed = proto.nodes_removed,
        nodes_updated = proto.nodes_updated,
        edges_added   = proto.edges_added,
        edges_removed = proto.edges_removed,
        edges_updated = proto.edges_updated,
    )


# ─── Communities ────────────────────────────────────────────────────────────


def _to_community_ref(proto: Any) -> CommunityRef:
    return CommunityRef(level=proto.level, community_id=proto.community_id)


def to_community_hierarchy(proto: Any) -> CommunityHierarchy:
    """Decodes a ``DetectCommunitiesResponse``, which both detection and the stored read return."""
    return CommunityHierarchy(
        communities = tuple(
            CommunityResult(
                id=each.id, level=each.level, parent_id=each.parent_id, size=each.size,
                member_node_ids=tuple(each.member_node_ids),
            )
            for each in proto.communities
        ),
        levels      = proto.levels,
        algorithm   = proto.algorithm,
        detected_at = proto.detected_at,
        stale_communities = tuple(
            _to_community_ref(each) for each in getattr(proto, "stale_communities", ())
        ),
    )


def to_community_summaries(proto: Any) -> CommunitySummaries:
    return CommunitySummaries(
        summaries=tuple(
            CommunitySummary(
                level=each.level, community_id=each.community_id,
                # Empty is "no report written", which client-java reads as null; see CommunitySummary.
                summary_text=each.summary_text or None,
            )
            for each in proto.summaries
        ),
        truncated=proto.truncated,
    )


def to_community_member_sets(proto: Any) -> tuple[CommunityMemberSet, ...]:
    return tuple(
        CommunityMemberSet(
            level=each.level, community_id=each.community_id,
            member_node_ids=tuple(each.member_node_ids), truncated=each.truncated,
        )
        for each in proto.communities
    )


def to_community_assignments(proto: Any) -> CommunityAssignments:
    return CommunityAssignments(
        by_node={each.node_id: each.community_id for each in proto.assignments},
        levels=proto.levels,
    )


def to_community_status(proto: Any) -> CommunityStatus:
    return CommunityStatus(
        detected        = proto.detected,
        detected_at     = proto.detected_at,
        stale           = proto.stale,
        levels          = proto.levels,
        community_count = proto.community_count,
        summary_count   = proto.summary_count,
    )


def to_global_search_results(proto: Any) -> tuple[GlobalSearchResult, ...]:
    return tuple(
        GlobalSearchResult(
            community_id=each.community_id, level=each.level, score=each.score,
            summary_text=each.summary_text, parent_community_id=each.parent_community_id,
            member_node_ids=tuple(each.member_node_ids),
        )
        for each in proto.results
    )


def to_refresh_communities_result(proto: Any) -> RefreshCommunitiesResult:
    return RefreshCommunitiesResult(
        hierarchy=to_community_hierarchy(proto.hierarchy),
        pruned_summaries=proto.pruned_summaries,
        stale_communities=tuple(_to_community_ref(each) for each in proto.stale_communities),
    )


# ─── Agentic memory ─────────────────────────────────────────────────────────


def to_observe_result(proto: Any) -> ObserveResult:
    """Decodes an ``ObserveResponse``.

    The wire field is ``linked_ids`` and the domain name is ``linked_edge_ids``, as in Java: they are
    edge ids, and the shorter wire name has been read as node ids before.
    """
    return ObserveResult(node_id=proto.node_id, linked_edge_ids=tuple(proto.linked_ids))


def to_decay_result(proto: Any) -> DecayResult:
    return DecayResult(pruned_nodes=proto.pruned_nodes, pruned_edges=proto.pruned_edges)


def to_edges(proto: Any) -> tuple[Edge, ...]:
    return tuple(to_edge(each) for each in proto.edges)


# ─── Natural-language search ────────────────────────────────────────────────


def to_nl_search_result(proto: Any) -> NlSearchResult:
    return NlSearchResult(
        matches=tuple(
            NlMatch(
                id           = each.id,
                score        = each.score,
                metadata     = from_struct(each.metadata),
                external_key = each.external_key if each.HasField("external_key") else None,
            )
            for each in proto.matches
        ),
        translated_query=proto.translated_query,
        query_type=proto.query_type,
    )


# ─── Change Data Capture ────────────────────────────────────────────────────


def _enum_or_unspecified(enum_type: Any, name: str) -> Any:
    """Reads a wire enum by name, falling back to UNSPECIFIED for one this build does not know.

    Unlike the statement enums, an unknown value here is not worth failing the stream over: a change
    of a kind this client cannot name is still a change at an LSN the consumer must checkpoint, and
    dropping the whole subscription would cost it every other event too.
    """
    try:
        return enum_type(name)
    except ValueError:
        return enum_type.UNSPECIFIED


def to_edge_info(proto: Any) -> EdgeInfo:
    """Decodes an ``EdgeInfo``, keeping a field the change did not carry as ``None``.

    Every field is ``optional`` on the wire precisely so absence can be told from a default: a weight update
    carries only the weight, a removed edge only its endpoints. Copying the defaults would report a weight
    update as an edge from node 0 to node 0, and node 0 is a real node.
    """
    return EdgeInfo(
        source_id = proto.source_id if proto.HasField("source_id") else None,
        target_id = proto.target_id if proto.HasField("target_id") else None,
        type      = proto.type if proto.HasField("type") else None,
        weight    = proto.weight if proto.HasField("weight") else None,
    )


def to_change_event(proto: Any) -> ChangeEvent:
    """Decodes a ``ChangeEvent``."""
    source = ChangeSource(
        resource_kind = proto.source.resource_kind,
        resource_id   = proto.source.resource_id,
        lsn           = proto.source.lsn,
        tx_id         = proto.source.tx_id,
        ts_ms         = proto.source.ts_ms,
    )
    if proto.snapshot_complete:
        # Every other field is unset on the marker; decoding them would invent defaults that read as
        # real values - an entity_id of 0, which is a valid id.
        return ChangeEvent(source=source, snapshot_complete=True)
    if proto.live:
        return ChangeEvent(source=source, live=True)

    from ._proto import cyrock_db_cdc_pb2 as cdc_pb2

    return ChangeEvent(
        source       = source,
        op           = _enum_or_unspecified(ChangeOp, cdc_pb2.ChangeOp.Name(proto.op)),
        entity_kind  = _enum_or_unspecified(EntityKind, cdc_pb2.EntityKind.Name(proto.entity_kind)),
        entity_id    = proto.entity_id,
        external_key = proto.external_key if proto.HasField("external_key") else None,
        labels       = tuple(proto.labels),
        metadata     = from_struct(proto.metadata) if proto.HasField("metadata") else {},
        vectors      = {name: tuple(v.values) for name, v in proto.vectors.items()},
        edge         = to_edge_info(proto.edge) if proto.HasField("edge") else None,
        schema       = (
            SchemaChange(
                added_fields=tuple(proto.schema.added_fields),
                dropped_fields=tuple(proto.schema.dropped_fields),
            )
            if proto.HasField("schema") else None
        ),
    )


# ─── Platform administration (CyrockDbAdminClient) ─────────────────────────
#
# As client-java's AdminProtoConverters: a protobuf string is "" when unset, so an optional id arrives as "" and is
# read as None; an unset timestamp is read as None rather than as the epoch.


_LATEST          = datetime.max.replace(tzinfo=timezone.utc)
_EARLIEST        = datetime.min.replace(tzinfo=timezone.utc)
# The seconds of the first and last instant a datetime holds, as integers: datetime.max.timestamp() is a float that
# rounds up to the first second of year 10000, the very value this has to catch.
_LATEST_SECONDS   = 253402300799
_EARLIEST_SECONDS = -62135596800


def _timestamp_or_none(proto: Any, name: str) -> datetime | None:
    """A timestamp field as an aware UTC datetime, ``None`` when unset.

    Clamped to ``datetime.max`` / ``datetime.min`` when the stored value lies beyond what a datetime can hold, as a
    key's expiry past year 9999 does when the platform stored it before it refused one: otherwise that one key would
    fail every listing of its organization. Java clamps to ``Instant.MAX`` / ``Instant.MIN`` the same way.
    """
    if not proto.HasField(name):
        return None
    timestamp = getattr(proto, name)
    if timestamp.seconds > _LATEST_SECONDS:
        return _LATEST
    if timestamp.seconds < _EARLIEST_SECONDS:
        return _EARLIEST
    moment: datetime = timestamp.ToDatetime(tzinfo=timezone.utc)
    return moment


def to_project_info(proto: Any) -> ProjectInfo:
    return ProjectInfo(
        id              = proto.id,
        organization_id = proto.organization_id,
        name            = proto.name,
        created_at      = _timestamp_or_none(proto, "created_at"),
        enabled         = proto.enabled,
    )


def to_project_infos(proto: Any) -> tuple[ProjectInfo, ...]:
    return tuple(to_project_info(each) for each in proto.projects)


def _api_key_status(name: str) -> ApiKeyStatus:
    """A key's status by name, or ``UNKNOWN`` for one this client does not know, as in Java.

    A status a newer platform added must not fail a whole listing, nor a revoke that has already happened.
    """
    status = ApiKeyStatus.__members__.get(name)
    return status if status is not None and status is not ApiKeyStatus.UNKNOWN else ApiKeyStatus.UNKNOWN


def to_api_key_info(proto: Any) -> ApiKeyInfo:
    return ApiKeyInfo(
        id              = proto.id,
        name            = proto.name,
        user_id         = proto.user_id or None,
        organization_id = proto.organization_id,
        project_id      = proto.project_id or None,
        permissions     = frozenset(proto.permissions),
        status          = _api_key_status(proto.status),
        created_at      = _timestamp_or_none(proto, "created_at"),
        expires_at      = _timestamp_or_none(proto, "expires_at"),
        last_used_at    = _timestamp_or_none(proto, "last_used_at"),
        revoked_at      = _timestamp_or_none(proto, "revoked_at"),
    )


def to_api_key_infos(proto: Any) -> tuple[ApiKeyInfo, ...]:
    return tuple(to_api_key_info(each) for each in proto.api_keys)


def to_created_api_key(proto: Any) -> CreatedApiKey:
    """The created key with its one-time secret.

    :raises CyrockDbClientException: with status 500 if the response carries no secret. The key was created
        regardless, so the message names it for revoking
    """
    api_key = to_api_key_info(proto.api_key)
    if not proto.plaintext_secret:
        raise CyrockDbClientException(
            500, f"The platform created API key {api_key.id} but returned no secret; revoke it and create another"
        )
    return CreatedApiKey(api_key=api_key, secret=proto.plaintext_secret)
