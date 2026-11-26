"""Vector compression: its default, its one validation rule, and that it survives the wire.

The enum's members and their values are held to the Java source by ``test_types_mirror_java``; what
is left to cover here is the defaulting, the dimension floor below which quantization saves nothing,
and the two mapping sites that carry the field to the server and back.
"""

from __future__ import annotations

from typing import Any

import pytest

from cyrock_db import CyrockDbClient
from cyrock_db._proto import cyrock_db_platform_pb2 as platform_pb2
from cyrock_db._proto_converters import _to_vector_field
from cyrock_db.types import (
    CollectionDefinition,
    GraphDefinition,
    VectorCompression,
    VectorFieldDefinition,
)


def _client(server: Any) -> CyrockDbClient:
    return CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key")


def test_vectorField_withoutCompression_defaultsToNone() -> None:
    field = VectorFieldDefinition(name="embedding", dimension=768)

    assert field.compression is VectorCompression.NONE


def test_vectorField_withCompressionNone_isAcceptedAtAnyDimension() -> None:
    # The floor applies to quantization only; full precision is the default and has no minimum.
    assert VectorFieldDefinition(name="tiny", dimension=2).compression is VectorCompression.NONE


def test_vectorField_withNvqAtAWorkableDimension_keepsIt() -> None:
    field = VectorFieldDefinition(name="embedding", dimension=768, compression=VectorCompression.NVQ)

    assert field.compression is VectorCompression.NVQ


@pytest.mark.parametrize("dimension", [1, 4, 10])
def test_vectorField_withNvqBelowTheFloor_raisesValueError(dimension: int) -> None:
    # A quantized vector costs 4 + dimension + 28 bytes against dimension * 4, so the two meet at
    # ten dimensions. Asking for it below eleven is a mistake worth reporting rather than honouring.
    with pytest.raises(ValueError) as raised:
        VectorFieldDefinition(name="embedding", dimension=dimension, compression=VectorCompression.NVQ)

    assert "embedding" in str(raised.value), "the message has to name the field it is about"
    assert "11" in str(raised.value)


def test_vectorField_withNvqAtTheFloor_isAccepted() -> None:
    field = VectorFieldDefinition(name="embedding", dimension=11, compression=VectorCompression.NVQ)

    assert field.compression is VectorCompression.NVQ


def test_createCollection_withCompression_sendsItAndReadsItBack(token_server: Any) -> None:
    definition = CollectionDefinition(
        name="docs",
        vector_fields=(
            VectorFieldDefinition(name="embedding", dimension=768, compression=VectorCompression.NVQ),
        ),
    )

    server = token_server()
    with _client(server) as client:
        created = client.create_collection("project-1", definition)

    sent = server.collections.requests[-1]
    assert sent.vector_fields[0].compression == "NVQ", "the name has to reach the wire"
    assert created.vector_fields[0].compression is VectorCompression.NVQ


def test_createGraph_withCompression_sendsItAndReadsItBack(token_server: Any) -> None:
    definition = GraphDefinition(
        name="kg",
        node_vector_fields=(
            VectorFieldDefinition(name="embedding", dimension=768, compression=VectorCompression.NVQ),
        ),
    )

    server = token_server()
    with _client(server) as client:
        created = client.create_graph("project-1", definition)

    sent = server.graph_defs.requests[-1]
    assert sent.node_vector_fields[0].compression == "NVQ"
    assert created.node_vector_fields[0].compression is VectorCompression.NVQ


def test_aDefinitionFromAnOlderServer_readsBackAsTheDefault() -> None:
    # Against the converter rather than through the in-process server: that server echoes the request
    # back, and the request builder always sends a compression, so a round trip never produces the empty
    # string an older server would - the compatibility branch would go untested.
    proto = platform_pb2.VectorFieldDefinitionProto(
        name="embedding",
        dimension=768,
        similarity_function="COSINE",
    )
    assert proto.compression == "", "the fixture has to be what an older server sends: nothing"

    field = _to_vector_field(proto)

    assert field.compression is VectorCompression.NONE


def test_aDefinitionCarryingACompression_readsItBack() -> None:
    # The other side of the same branch, so the test above cannot pass by the converter ignoring the field.
    proto = platform_pb2.VectorFieldDefinitionProto(
        name="embedding",
        dimension=768,
        similarity_function="COSINE",
        compression="NVQ",
    )

    assert _to_vector_field(proto).compression is VectorCompression.NVQ
