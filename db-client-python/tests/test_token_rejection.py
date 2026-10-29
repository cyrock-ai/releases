"""A client recovers from a token the server stops accepting.

The scenario is a server restart that changes the signing key, which the all-in-one image does on every
start: the cached token is still unexpired, but no longer valid. The rejected call fails; the one after it
must carry a fresh token. The exchange count is what tells a recovered client from one that keeps sending
the rejected token until it expires. The counterpart of ``client-java``'s ``TokenRejectionInterceptorTest``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import grpc
import pytest

from cyrock_db import CyrockDbClient
from cyrock_db._auth import AsyncAuthInterceptor, AuthInterceptor, Metadata, bearer
from cyrock_db._proto import cyrock_db_cdc_pb2 as cdc_pb2
from cyrock_db._proto import cyrock_db_cdc_pb2_grpc as cdc_grpc
from cyrock_db.aio import AsyncCyrockDbClient
from cyrock_db.exceptions import UnauthorizedException

COLLECTION = "collection-1"


def _restart(server: Any) -> None:
    """What the client sees of a restart with a new signing key: new tokens, and only those accepted."""
    server.service.token       = "jwt-2"
    server.docs.accepted_token = "jwt-2"


def test_list_tokenRejectedAfterARestart_nextCallExchangesAFreshToken(token_server: Any) -> None:
    server = token_server(token="jwt-1", docs={"accepted_token": "jwt-1"})
    with CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key") as client:
        client.list(COLLECTION, 0, 10)
        _restart(server)

        with pytest.raises(UnauthorizedException):
            client.list(COLLECTION, 0, 10)
        client.list(COLLECTION, 0, 10)

    assert len(server.service.requests) == 2


def test_listCollections_apiKeyRejected_keepsTheToken(token_server: Any) -> None:
    """A definition service authenticates with the API key, so its rejection says nothing about the token."""
    server = token_server(token="jwt-1")
    with CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key") as client:
        client.list(COLLECTION, 0, 10)
        server.collections.fail_with = grpc.StatusCode.UNAUTHENTICATED

        with pytest.raises(UnauthorizedException):
            client.list_collections("project-1")
        client.list(COLLECTION, 0, 10)

    assert len(server.service.requests) == 1


async def test_asyncList_tokenRejectedAfterARestart_nextCallExchangesAFreshToken(token_server: Any) -> None:
    server = token_server(token="jwt-1", docs={"accepted_token": "jwt-1"})
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key") as client:
        await client.list(COLLECTION, 0, 10)
        _restart(server)

        with pytest.raises(UnauthorizedException):
            await client.list(COLLECTION, 0, 10)
        await client.list(COLLECTION, 0, 10)

    assert len(server.service.requests) == 2


# ─── Streams ──────────────────────────────────────────────────────────────────────────────────────
#
# The change stream is the one streaming call. Its reconnect counts on fetching a fresh token, so a stream
# the server rejects has to report its credential like a unary call does. The report arrives when the
# stream ends rather than before the caller sees the error, hence the waits.

REPORT_TIMEOUT_SECONDS = 5.0


def test_syncInterceptor_streamEndingUnauthenticated_reportsItsCredential(token_server: Any) -> None:
    server   = token_server(cdc={"batches": [grpc.StatusCode.UNAUTHENTICATED]})
    reported = threading.Event()
    rejected: list[Metadata] = []

    def on_rejected(credential: Metadata) -> None:
        rejected.append(credential)
        reported.set()

    channel = grpc.intercept_channel(
        server.channel, AuthInterceptor(lambda method: [bearer("jwt-1")], on_rejected=on_rejected)
    )
    stream = cdc_grpc.ChangeStreamGrpcServiceStub(channel).WatchChanges(cdc_pb2.WatchRequest())  # type: ignore[no-untyped-call]
    with pytest.raises(grpc.RpcError):
        list(stream)

    assert reported.wait(REPORT_TIMEOUT_SECONDS)
    assert rejected == [[bearer("jwt-1")]]


async def test_asyncInterceptor_streamEndingUnauthenticated_reportsItsCredential(token_server: Any) -> None:
    server   = token_server(cdc={"batches": [grpc.StatusCode.UNAUTHENTICATED]})
    reported = asyncio.Event()
    rejected: list[Metadata] = []

    async def credential(method: str) -> list[tuple[str, str]]:
        return [bearer("jwt-1")]

    def on_rejected(credential: Metadata) -> None:
        rejected.append(credential)
        reported.set()

    interceptors = AsyncAuthInterceptor(credential, on_rejected=on_rejected).interceptors()
    async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}", interceptors=interceptors) as channel:
        stream = cdc_grpc.ChangeStreamGrpcServiceStub(channel).WatchChanges(cdc_pb2.WatchRequest())  # type: ignore[no-untyped-call]
        with pytest.raises(grpc.aio.AioRpcError):
            async for _ in stream:
                pass

        await asyncio.wait_for(reported.wait(), REPORT_TIMEOUT_SECONDS)

    assert rejected == [[bearer("jwt-1")]]
