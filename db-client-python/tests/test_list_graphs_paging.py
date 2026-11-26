"""The paged graph and collection listings on both facades (#1147), as client-java's ``listGraphsPage``.

``list_graphs_page`` sends the page size and token and maps the next one back; ``list_graphs`` follows the tokens to
the last page, also against a server that predates paging and answers everything at once.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cyrock_db import CyrockDbClient
from cyrock_db import _proto_converters as converters
from cyrock_db import _proto_requests as requests
from cyrock_db._proto import cyrock_db_platform_pb2 as platform_pb2
from cyrock_db.aio import AsyncCyrockDbClient
from cyrock_db.types import Page

PROJECT = "project-1"


def _client(server: Any) -> CyrockDbClient:
    return CyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key")


def _async_client(server: Any) -> AsyncCyrockDbClient:
    return AsyncCyrockDbClient(host="127.0.0.1", port=server.port, api_key="api-key")


def _seed_graphs(server: Any, count: int) -> list[str]:
    ids = [f"g{index:05d}" for index in range(count)]
    server.graph_defs.graphs.extend(platform_pb2.GraphDefinitionProto(id=each, name=each) for each in ids)
    return ids


def _seed_collections(server: Any, count: int) -> list[str]:
    ids = [f"c{index:05d}" for index in range(count)]
    server.collections.collections.extend(
        platform_pb2.CollectionDefinitionProto(id=each, name=each) for each in ids
    )
    return ids


def test_listForProject_pageSizeAndToken_areSent() -> None:
    first = requests.list_for_project(PROJECT, 50, None)
    nxt   = requests.list_for_project(PROJECT, 50, "abc")

    assert first.project_id == PROJECT
    assert first.HasField("page_size")
    assert first.page_size == 50
    assert first.page_token == ""
    assert nxt.page_token == "abc"


def test_toGraphDefinitionPage_response_carriesTheItemsAndTheNextPageToken() -> None:
    response = platform_pb2.GraphDefinitionList(
        graphs=[platform_pb2.GraphDefinitionProto(id="a", name="A")], next_page_token="next"
    )

    page = converters.to_graph_definition_page(response)

    assert [each.id for each in page.items] == ["a"]
    assert page.next_page_token == "next"
    assert page.has_next_page()


def test_toGraphDefinitionPage_lastPage_hasNoNextPage() -> None:
    page = converters.to_graph_definition_page(platform_pb2.GraphDefinitionList())

    assert page.items == ()
    assert not page.has_next_page()


def test_page_itemsNotATuple_raisesValueError() -> None:
    with pytest.raises(ValueError, match="items must be a tuple"):
        Page(items=["a"])  # type: ignore[arg-type]


def test_listGraphsPage_firstPage_returnsThePageAndTheTokenOfTheNext(token_server: Any) -> None:
    server = token_server()
    ids    = _seed_graphs(server, 5)
    with _client(server) as client:
        page = client.list_graphs_page(PROJECT, 2)

    assert [each.id for each in page.items] == ids[:2]
    assert page.next_page_token == "2"
    assert server.graph_defs.requests[-1].page_size == 2


def test_listGraphs_moreThanOnePage_followsTheTokensToTheLastPage(token_server: Any) -> None:
    server = token_server()
    ids    = _seed_graphs(server, requests.LISTING_PAGE_SIZE * 2 + 1)
    with _client(server) as client:
        graphs = client.list_graphs(PROJECT)

    assert [each.id for each in graphs] == ids
    sent = server.graph_defs.requests
    assert [each.page_token for each in sent] == ["", "1000", "2000"]
    assert all(each.page_size == requests.LISTING_PAGE_SIZE for each in sent)


def test_listGraphs_serverThatPredatesPaging_takesEverythingFromTheOneResponse(token_server: Any) -> None:
    server = token_server()
    server.graph_defs.ignores_paging = True
    ids    = _seed_graphs(server, requests.LISTING_PAGE_SIZE + 10)
    with _client(server) as client:
        graphs = client.list_graphs(PROJECT)

    assert len(graphs) == len(ids)
    assert len(server.graph_defs.requests) == 1


def test_listGraphs_emptyProject_returnsAnEmptyTupleAfterOneCall(token_server: Any) -> None:
    server = token_server()
    with _client(server) as client:
        assert client.list_graphs(PROJECT) == ()

    assert len(server.graph_defs.requests) == 1


def test_listCollections_moreThanOnePage_followsTheTokensToTheLastPage(token_server: Any) -> None:
    server = token_server()
    ids    = _seed_collections(server, requests.LISTING_PAGE_SIZE + 1)
    with _client(server) as client:
        first       = client.list_collections_page(PROJECT, 2, "")
        collections = client.list_collections(PROJECT)

    assert [each.id for each in first.items] == ids[:2]
    assert first.has_next_page()
    assert [each.id for each in collections] == ids


def test_asyncListGraphs_moreThanOnePage_followsTheTokensToTheLastPage(token_server: Any) -> None:
    server = token_server()
    ids    = _seed_graphs(server, requests.LISTING_PAGE_SIZE + 1)

    async def run() -> tuple[Page[Any], tuple[Any, ...]]:
        async with _async_client(server) as client:
            return await client.list_graphs_page(PROJECT, 3), await client.list_graphs(PROJECT)

    page, graphs = asyncio.run(run())

    assert [each.id for each in page.items] == ids[:3]
    assert page.next_page_token == "3"
    assert [each.id for each in graphs] == ids


def test_asyncListCollections_moreThanOnePage_followsTheTokensToTheLastPage(token_server: Any) -> None:
    server = token_server()
    ids    = _seed_collections(server, requests.LISTING_PAGE_SIZE + 1)

    async def run() -> tuple[Page[Any], tuple[Any, ...]]:
        async with _async_client(server) as client:
            return await client.list_collections_page(PROJECT, 1), await client.list_collections(PROJECT)

    page, collections = asyncio.run(run())

    assert [each.id for each in page.items] == ids[:1]
    assert [each.id for each in collections] == ids


def _seed_unordered(server: Any) -> None:
    """Seeds graphs and collections as a server that predates paging holds them: in hash-map order."""
    server.graph_defs.ignores_paging = True
    server.graph_defs.graphs.extend(
        platform_pb2.GraphDefinitionProto(id=each, name=each) for each in ("g3", "g1", "g10", "g2")
    )
    server.collections.collections.extend(
        platform_pb2.CollectionDefinitionProto(id=each, name=each) for each in ("c2", "c10", "c1")
    )


def test_listGraphs_unorderedAnswerOfAServerThatPredatesPaging_returnsAscendingIdOrder(token_server: Any) -> None:
    server = token_server()
    _seed_unordered(server)
    with _client(server) as client:
        graphs      = client.list_graphs(PROJECT)
        collections = client.list_collections(PROJECT)

    # String order, as the server pages by: "g10" sorts before "g2".
    assert [each.id for each in graphs] == ["g1", "g10", "g2", "g3"]
    assert [each.id for each in collections] == ["c1", "c10", "c2"]


def test_asyncListGraphs_unorderedAnswerOfAServerThatPredatesPaging_returnsAscendingIdOrder(
    token_server: Any,
) -> None:
    server = token_server()
    _seed_unordered(server)

    async def run() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        async with _async_client(server) as client:
            return await client.list_graphs(PROJECT), await client.list_collections(PROJECT)

    graphs, collections = asyncio.run(run())

    assert [each.id for each in graphs] == ["g1", "g10", "g2", "g3"]
    assert [each.id for each in collections] == ["c1", "c10", "c2"]


def test_sortedById_idsOutsideTheBasicPlane_followJavaStringOrder() -> None:
    # Java's String.compareTo compares UTF-16 code units: U+1F600 is a surrogate pair from 0xD83D, which sorts
    # before U+FF01, although its code point is larger. Python's own str order would put it last.
    emoji, fullwidth = "\U0001f600", "！"
    graphs = tuple(
        converters.to_graph_definition(platform_pb2.GraphDefinitionProto(id=each, name="n"))
        for each in (fullwidth, emoji)
    )

    assert [each.id for each in converters.sorted_by_id(graphs)] == [emoji, fullwidth]
