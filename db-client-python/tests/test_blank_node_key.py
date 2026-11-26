"""Node upsert external-key presence on the wire (#510).

A blank external key is refused server-side, and a keyless upsert must stay keyless. The client's job is
to set ``external_key`` on the request only when the caller supplied one: an omitted (``None``) key leaves
the field absent (keyless create), while a present key - blank included - sets field presence so the server
routes it to the by-key path and rejects a blank one, mirroring ``client-java``.
"""

from __future__ import annotations

from typing import Any

import pytest

from cyrock_db import CyrockDbClient
from cyrock_db.types import UpsertNodeRequest

GRAPH = "graph-1"


def _client(server: Any) -> CyrockDbClient:
    return CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key")


def test_upsertNode_omitsExternalKeyWhenNone_soAKeylessUpsertStaysKeyless(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        client.upsert_node(GRAPH, UpsertNodeRequest(labels=["N"], external_key=None))

    sent = server.graph.requests[-1]
    assert not sent.HasField("external_key"), "a None key must be absent on the wire (keyless upsert)"


@pytest.mark.parametrize("key", ["", "   ", "k-1"])
def test_upsertNode_setsExternalKeyPresenceWhenProvided(token_server: Any, key: str) -> None:
    server = token_server()
    with _client(server) as client:
        client.upsert_node(GRAPH, UpsertNodeRequest(labels=["N"], external_key=key))

    sent = server.graph.requests[-1]
    assert sent.HasField("external_key"), "a provided key (blank included) must carry presence"
    assert sent.external_key == key
