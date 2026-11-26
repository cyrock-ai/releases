"""The keyword classifier skips exactly what the CyQL lexer skips: whitespace and ``--`` line comments.

``tests/test_cyql_routing.py`` holds the classifier to the parser's fixture, but every fixture entry is
one line without a comment, so it cannot see how the leading keywords are found. The grammar's
``LINE_COMMENT`` is ``'--' ~[\\r\\n]*`` and it has no other comment form. A router that skips some
other form, or misses this one, sends a commented ``CREATE GRAPH`` to the data plane, which refuses it.

These need no fixture, so unlike that module they run in a published copy too.
"""

from __future__ import annotations

import pytest

from cyrock_db._cyql_routing import is_project_scoped, statement_class
from cyrock_db.types import StatementClass


@pytest.mark.parametrize(
    "statement",
    [
        "-- note\nCREATE GRAPH g",
        "-- first\n-- second\r\n  CREATE GRAPH g",
        "CREATE -- the kind follows\nGRAPH g",
        "\t\r\n--\nSHOW GRAPHS",
        "DROP GRAPH-- trailing\n",
        "describe collection c -- lower case",
    ],
)
def test_isProjectScoped_leadingOrInterleavedLineComments_areSkipped(statement: str) -> None:
    assert is_project_scoped(statement) is True


@pytest.mark.parametrize(
    "statement",
    [
        "// not a CyQL comment\nCREATE GRAPH g",
        "/* not a CyQL comment */ CREATE GRAPH g",
        "-- CREATE GRAPH g",
        "CREATE (n:Label)",
        "CREATE(n:Label)",
        "-- note\nCREATE (n:Label)",
        "",
    ],
)
def test_isProjectScoped_otherCommentFormsOrADataWrite_isNotProjectScoped(statement: str) -> None:
    """``//`` and ``/* */`` are not CyQL; the server, not the router, reports them as a syntax error."""
    assert is_project_scoped(statement) is False


def test_statementClass_lineCommentBeforeAlter_classifiesByTheKeyword() -> None:
    assert statement_class("-- migrate\nALTER GRAPH g ADD NODE FIELD x STRING") is StatementClass.GRAPH_SCOPED_DDL
