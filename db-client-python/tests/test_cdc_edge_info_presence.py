"""The EdgeInfo of a change event says which fields the change carried (#1105).

Every field of ``EdgeInfo`` is ``optional`` in ``cyrock_db_cdc.proto``: "Fields are absent when the originating
change did not carry them". A weight update carries the weight only, a removed edge its endpoints only. Node 0
is a real node, so an absent endpoint must come out as ``None``, not as 0, and an absent type or weight as
``None``, not as ``''`` or ``0.0``. Java tells the same apart with ``hasSourceId()`` and its siblings.
"""

from __future__ import annotations

from typing import Any

from cyrock_db import _proto_converters as converters
from cyrock_db._proto import cyrock_db_cdc_pb2 as cdc_pb2
from cyrock_db.types import ChangeOp, EdgeInfo, EntityKind


def _edge_event(op: Any, edge: Any) -> Any:
    return cdc_pb2.ChangeEvent(
        source=cdc_pb2.ChangeSource(resource_id="graph-1", lsn=3),
        op=op,
        entity_kind=cdc_pb2.EntityKind.EDGE,
        entity_id=7,
        edge=edge,
    )


def test_toChangeEvent_weightOnlyEdgeUpdate_leavesEndpointsAndTypeUnset() -> None:
    proto = _edge_event(cdc_pb2.ChangeOp.UPDATE, cdc_pb2.EdgeInfo(weight=0.5))

    event = converters.to_change_event(proto)

    assert event.op is ChangeOp.UPDATE
    assert event.entity_kind is EntityKind.EDGE
    assert event.edge == EdgeInfo(source_id=None, target_id=None, type=None, weight=0.5)


def test_toChangeEvent_removedEdge_carriesEndpointsOnly() -> None:
    proto = _edge_event(cdc_pb2.ChangeOp.DELETE, cdc_pb2.EdgeInfo(source_id=1, target_id=2))

    event = converters.to_change_event(proto)

    assert event.edge == EdgeInfo(source_id=1, target_id=2, type=None, weight=None)


def test_toChangeEvent_addedEdge_carriesEveryField() -> None:
    proto = _edge_event(
        cdc_pb2.ChangeOp.CREATE, cdc_pb2.EdgeInfo(source_id=1, target_id=2, type="KNOWS", weight=0.75)
    )

    event = converters.to_change_event(proto)

    assert event.edge == EdgeInfo(source_id=1, target_id=2, type="KNOWS", weight=0.75)


def test_toChangeEvent_explicitZeroAndEmptyValues_areKeptAsSent() -> None:
    """Presence, not truthiness: an edge from node 0 with an empty type and weight 0 is a real edge."""
    proto = _edge_event(
        cdc_pb2.ChangeOp.CREATE, cdc_pb2.EdgeInfo(source_id=0, target_id=0, type="", weight=0.0)
    )

    event = converters.to_change_event(proto)

    assert event.edge == EdgeInfo(source_id=0, target_id=0, type="", weight=0.0)


def test_toChangeEvent_edgeMetadataUpdate_hasNoEdgeInfo() -> None:
    proto = cdc_pb2.ChangeEvent(op=cdc_pb2.ChangeOp.UPDATE, entity_kind=cdc_pb2.EntityKind.EDGE, entity_id=7)

    assert converters.to_change_event(proto).edge is None


def test_edgeInfo_defaults_areAllUnset() -> None:
    assert EdgeInfo() == EdgeInfo(source_id=None, target_id=None, type=None, weight=None)


def test_toChangeEvent_absentExternalKey_isNoneAndAPresentEmptyOneIsKept() -> None:
    """``external_key`` is the other ``optional`` field the converter maps; it already honours presence."""
    absent  = cdc_pb2.ChangeEvent(op=cdc_pb2.ChangeOp.UPSERT, entity_kind=cdc_pb2.EntityKind.DOCUMENT)
    present = cdc_pb2.ChangeEvent(
        op=cdc_pb2.ChangeOp.UPSERT, entity_kind=cdc_pb2.EntityKind.DOCUMENT, external_key=""
    )

    assert converters.to_change_event(absent).external_key is None
    assert converters.to_change_event(present).external_key == ""
