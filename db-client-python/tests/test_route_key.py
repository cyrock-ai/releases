"""Every data-plane call carries the store it is about as the route key header.

The counterpart of ``client-java``'s ``RouteKeyHeaderTest`` and ``RouteKeyClientInterceptorTest``: the header the
gateway routes on, see ``cyrock_db._route_key``, arrives on data-plane calls in both facades and never on platform
calls, and every data-plane request names its store in one of the fields the interceptor looks for.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cyrock_db import CyrockDbClient
from cyrock_db._proto import cyrock_db_cdc_pb2 as cdc_pb2
from cyrock_db._proto import cyrock_db_data_pb2 as data_pb2
from cyrock_db._proto import cyrock_db_graph_pb2 as graph_pb2
from cyrock_db._proto import cyrock_db_nlsearch_pb2 as nlsearch_pb2
from cyrock_db._route_key import (
    DATA_PLANE_PREFIX,
    MAX_LENGTH,
    ROUTE_KEY_HEADER,
    STORE_FIELDS,
    RouteKeyInterceptor,
    route_key_for,
    route_key_of,
)
from cyrock_db.aio import AsyncCyrockDbClient

COLLECTION = "collection-1"

DATA_PLANE_FILES = (data_pb2, graph_pb2, nlsearch_pb2, cdc_pb2)


def test_list_anyClient_sendsTheCollectionAsRouteKey(token_server: Any) -> None:
    server = token_server()
    with CyrockDbClient(host="127.0.0.1", port=server.port) as client:
        client.list(COLLECTION, 0, 10)

    assert (ROUTE_KEY_HEADER, COLLECTION) in server.docs.metadata


async def test_asyncList_anyClient_sendsTheCollectionAsRouteKey(token_server: Any) -> None:
    server = token_server()
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port) as client:
        await client.list(COLLECTION, 0, 10)

    assert (ROUTE_KEY_HEADER, COLLECTION) in server.docs.metadata


def test_listCollections_platformCall_sendsNoRouteKey(token_server: Any) -> None:
    server = token_server()
    with CyrockDbClient(host="127.0.0.1", port=server.port) as client:
        client.list_collections("project-1")

    assert ROUTE_KEY_HEADER not in dict(server.collections.metadata)


def test_routeKeyOf_everyDataPlaneRequest_namesItsStore() -> None:
    for module in DATA_PLANE_FILES:
        for service in module.DESCRIPTOR.services_by_name.values():
            assert f"/{service.full_name}/".startswith(DATA_PLANE_PREFIX)
            for method in service.methods:
                fields = method.input_type.fields_by_name
                assert any(name in fields for name in STORE_FIELDS), (
                    f"{service.full_name}.{method.name} names no store in any of {STORE_FIELDS}: name it with one of"
                    " them, or add its field there and in Java's RouteKeyHeader.STORE_FIELDS"
                )


def test_routeKeyOf_mergeRequest_isTheBranch() -> None:
    request = graph_pb2.MergeGraphDataRequest(branch_graph_id="branch", parent_graph_id="parent")

    assert route_key_of(request) == "branch"


def test_routeKeyOf_watchRequest_isTheResource() -> None:
    assert route_key_of(cdc_pb2.WatchRequest(resource_id="graph-x", from_now=True)) == "graph-x"


@pytest.mark.parametrize("store", ["", "küche", "line\nbreak", "x" * (MAX_LENGTH + 1)])
def test_routeKeyOf_invalidHeaderValue_isNone(store: str) -> None:
    assert route_key_of(data_pb2.GetDocumentRequest(collection_id=store)) is None


def test_routeKeyOf_longestValidValue_isKept() -> None:
    store = "x" * MAX_LENGTH

    assert route_key_of(data_pb2.GetDocumentRequest(collection_id=store)) == store


def test_interceptUnaryUnary_headerSetByCaller_keepsIt() -> None:
    sent: list[Any] = []
    details = SimpleNamespace(
        method         = "/ai.cyrock.db.data.server.DocumentGrpcService/GetById",
        timeout        = None,
        metadata       = ((ROUTE_KEY_HEADER, "chosen"),),
        credentials    = None,
        wait_for_ready = None,
        compression    = None,
    )

    RouteKeyInterceptor().intercept_unary_unary(
        lambda call_details, _request: sent.append(call_details),
        details,  # type: ignore[arg-type]
        data_pb2.GetDocumentRequest(collection_id=COLLECTION),
    )

    assert list(sent[0].metadata) == [(ROUTE_KEY_HEADER, "chosen")]


def test_routeKeyOf_branchRequest_isTheRoot() -> None:
    assert route_key_of(graph_pb2.GetNodeRequest(graph_id="root-1~branch-a")) == "root-1"


@pytest.mark.parametrize(
    ("store_id", "expected"),
    [("root-1", "root-1"), ("root-1~branch-a", "root-1"), ("root-1~a~b", "root-1"), ("~leading", "~leading")],
)
def test_routeKeyFor_storeIds_isThePartBeforeTheFirstSeparator(store_id: str, expected: str) -> None:
    assert route_key_for(store_id) == expected
