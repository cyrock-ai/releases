"""A client created with ``consistent_reads=True`` asks for a consistent read on every call.

The counterpart of ``client-java``'s ``ConsistentReadHeaderTest``: the header the data server reads, see
``cyrock_db._consistency``, arrives on the call, in both facades, and is absent by default.
"""

from __future__ import annotations

from typing import Any

from cyrock_db import CyrockDbClient
from cyrock_db._consistency import CONSISTENT, CONSISTENT_READ_HEADER
from cyrock_db.aio import AsyncCyrockDbClient

COLLECTION = "collection-1"


def test_list_consistentReads_sendsTheConsistentReadHeader(token_server: Any) -> None:
    server = token_server()
    with CyrockDbClient(host="127.0.0.1", port=server.port, consistent_reads=True) as client:
        client.list(COLLECTION, 0, 10)

    assert (CONSISTENT_READ_HEADER, CONSISTENT) in server.docs.metadata


def test_list_consistentReadsWithAnApiKey_sendsTheHeaderAndTheToken(token_server: Any) -> None:
    server = token_server(token="jwt-1")
    with CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key", consistent_reads=True) as client:
        client.list(COLLECTION, 0, 10)

    assert (CONSISTENT_READ_HEADER, CONSISTENT) in server.docs.metadata
    assert ("authorization", "Bearer jwt-1") in server.docs.metadata


def test_list_default_sendsNoConsistentReadHeader(token_server: Any) -> None:
    server = token_server()
    with CyrockDbClient(host="127.0.0.1", port=server.port) as client:
        client.list(COLLECTION, 0, 10)

    assert CONSISTENT_READ_HEADER not in dict(server.docs.metadata)


async def test_asyncList_consistentReads_sendsTheConsistentReadHeader(token_server: Any) -> None:
    server = token_server()
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, consistent_reads=True) as client:
        await client.list(COLLECTION, 0, 10)

    assert (CONSISTENT_READ_HEADER, CONSISTENT) in server.docs.metadata


async def test_asyncList_consistentReadsWithAnApiKey_sendsTheHeaderAndTheToken(token_server: Any) -> None:
    server = token_server(token="jwt-1")
    async with AsyncCyrockDbClient(
        host="127.0.0.1", port=server.port, api_key="api-key", consistent_reads=True
    ) as client:
        await client.list(COLLECTION, 0, 10)

    assert (CONSISTENT_READ_HEADER, CONSISTENT) in server.docs.metadata
    assert ("authorization", "Bearer jwt-1") in server.docs.metadata


async def test_asyncList_default_sendsNoConsistentReadHeader(token_server: Any) -> None:
    server = token_server()
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port) as client:
        await client.list(COLLECTION, 0, 10)

    assert CONSISTENT_READ_HEADER not in dict(server.docs.metadata)
