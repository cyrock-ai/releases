"""Holds the Python admin client's operation signatures to the Java reference admin client's.

``client-java/src/test/resources/admin-operation-signatures.tsv`` is generated from ``CyrockDbAdminClient`` by
``AdminOperationSignatureContractTest``; this holds :class:`~cyrock_db.CyrockDbAdminClient` to it, operation for
operation, parameter for parameter, in order - and in the other direction, so an operation Python has and Java
does not fails here as drift. Skipped when the reactor is not beside us, as ``test_operations_mirror_java.py`` is.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from cyrock_db import CyrockDbAdminClient

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "client-java" / "src" / "test" / "resources" / "admin-operation-signatures.tsv"
)

pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file(),
    reason="the client-java module is not beside this checkout; nothing to compare against",
)


def _rows() -> dict[str, list[str]]:
    if not FIXTURE.is_file():
        return {}
    rows = {}
    for line in FIXTURE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        operation, _, params = stripped.partition("\t")
        rows[operation] = [p for p in params.split(",") if p]
    return rows


def _python_operations() -> dict[str, list[str]]:
    return {
        name: [parameter for parameter in inspect.signature(member).parameters if parameter != "self"]
        for name, member in inspect.getmembers(CyrockDbAdminClient, predicate=inspect.isfunction)
        if not name.startswith("_") and name != "close"
    }


@pytest.mark.parametrize(("operation", "java_params"), sorted(_rows().items()))
def test_everyJavaAdminOperation_hasAPythonTwin_takingTheSameParametersInOrder(
    operation: str, java_params: list[str]
) -> None:
    python = _python_operations()
    assert operation in python, f"CyrockDbAdminClient (Python) has no '{operation}', which client-java offers"
    assert python[operation] == java_params, (
        f"'{operation}' takes different parameters in Python than in Java.\n"
        f"  Java:   {java_params}\n"
        f"  Python: {python[operation]}"
    )


def test_pythonAdminClient_hasNoOperationJavaLacks() -> None:
    extra = sorted(set(_python_operations()) - set(_rows()))
    assert not extra, (
        f"CyrockDbAdminClient (Python) has operations client-java does not: {extra}. Add them to Java first."
    )


def test_theFixture_isNotTrivial() -> None:
    assert len(_rows()) >= 6, "admin-operation-signatures.tsv looks empty or truncated"
