"""Filter parameters: the values bound to a search filter's ``$name`` references (issue #689).

These assert on the encoded request only. Binding happens on the server, which compares each value as a
literal; what the SDK owes is to carry the map to the wire intact and to refuse a malformed one early.
"""

from __future__ import annotations

from typing import Any

import pytest

from cyrock_db import _proto_requests as requests
from cyrock_db.types import (
    FusionStrategy,
    HybridSearchRequest,
    ListNodesRequest,
    MultiVectorSearchRequest,
    SearchRequest,
    SearchSimilarRequest,
    VectorQuery,
)

PAYLOAD = "nope' OR visibility IS NOT NULL OR visibility = 'x"


def test_search_filterParameters_areEncodedAsAStruct() -> None:
    built = requests.search("c1", SearchRequest(
        "embedding", (1.0,), 5, filter="visibility = $v", filter_parameters={"v": PAYLOAD},
    ))

    assert built.filter == "visibility = $v"
    assert built.filter_parameters.fields["v"].string_value == PAYLOAD


def test_search_noFilterParameters_leavesTheStructUnset() -> None:
    built = requests.search("c1", SearchRequest("embedding", (1.0,), 5, filter="i > 1"))

    assert not built.HasField("filter_parameters")


def test_search_integerAndListParameters_keepTheirTypes() -> None:
    built = requests.search("c1", SearchRequest(
        "embedding", (1.0,), 5,
        filter="year >= $y AND tag IN $t", filter_parameters={"y": 2020, "t": ["a", "b"]},
    ))

    assert built.filter_parameters.fields["y"].int_value == 2020
    assert len(built.filter_parameters.fields["t"].list_value.values) == 2


def test_multiSearch_filterParameters_areEncodedAsAStruct() -> None:
    built = requests.multi_search("c1", MultiVectorSearchRequest(
        (VectorQuery("embedding", (1.0,)),), 5, FusionStrategy.RRF,
        filter="owner = $o", filter_parameters={"o": "alice"},
    ))

    assert built.filter_parameters.fields["o"].string_value == "alice"


def test_hybridSearch_filterParameters_areEncodedAsAStruct() -> None:
    built = requests.hybrid_search("c1", HybridSearchRequest(
        vector_queries=(VectorQuery("embedding", (1.0,)),),
        filter="owner = $o", filter_parameters={"o": "alice"},
    ))

    assert built.filter_parameters.fields["o"].string_value == "alice"


def test_searchSimilar_filterParameters_areEncodedAsAStruct() -> None:
    built = requests.search_similar("g1", SearchSimilarRequest(
        "embedding", (1.0,), 5, filter="owner = $o", filter_parameters={"o": "alice"},
    ))

    assert built.filter_parameters.fields["o"].string_value == "alice"


def test_listNodes_filterParameters_areEncodedAsAStruct() -> None:
    built = requests.list_nodes("g1", ListNodesRequest(filter="owner = $o", filter_parameters={"o": "alice"}))

    assert built.filter_parameters.fields["o"].string_value == "alice"


def test_filterParameters_mutatingTheSource_doesNotChangeTheRequest() -> None:
    source  = {"v": "public"}
    request = SearchRequest("embedding", (1.0,), 5, filter="visibility = $v", filter_parameters=source)
    source["v"] = "private"

    assert request.filter_parameters == {"v": "public"}


def test_filterParameters_listValue_isFrozen() -> None:
    values  = ["a", "b"]
    request = SearchRequest("embedding", (1.0,), 5, filter="tag IN $t", filter_parameters={"t": values})
    values.append("c")

    assert request.filter_parameters == {"t": ("a", "b")}


@pytest.mark.parametrize("parameters", [{"": "x"}, {" ": "x"}, {"v": None}, {"v": ["a", None]}])
def test_filterParameters_malformed_raisesValueError(parameters: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SearchRequest("embedding", (1.0,), 5, filter="visibility = $v", filter_parameters=parameters)
