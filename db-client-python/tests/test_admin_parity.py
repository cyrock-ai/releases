"""Holds :class:`AsyncCyrockDbAdminClient` to :class:`CyrockDbAdminClient`.

The admin counterpart of ``test_parity.py``: the two admin facades are written separately, so an operation added to
one and forgotten on the other would ship unnoticed. Fails in both directions, and on any difference in parameters.
"""

from __future__ import annotations

import inspect
from typing import Any

from cyrock_db import CyrockDbAdminClient
from cyrock_db.aio import AsyncCyrockDbAdminClient


def _operations(client_type: type) -> dict[str, inspect.Signature]:
    """The public operations of a client: its own methods, minus dunders and context management."""
    return {
        name: inspect.signature(member)
        for name, member in inspect.getmembers(client_type, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def _comparable(signature: inspect.Signature) -> list[tuple[str, Any]]:
    """A signature reduced to what parity is about: parameter names and annotations, minus self."""
    return [(name, parameter.annotation) for name, parameter in signature.parameters.items() if name != "self"]


def test_asyncAdminClient_comparedToTheSyncOne_hasTheSameOperations() -> None:
    assert set(_operations(AsyncCyrockDbAdminClient)) == set(_operations(CyrockDbAdminClient))


def test_everyAdminTwin_takesTheSameParameters() -> None:
    synchronous  = _operations(CyrockDbAdminClient)
    asynchronous = _operations(AsyncCyrockDbAdminClient)

    mismatched = [
        f"{name}: sync {_comparable(signature)} vs async {_comparable(asynchronous[name])}"
        for name, signature in synchronous.items()
        if _comparable(signature) != _comparable(asynchronous[name])
    ]
    assert not mismatched, "Admin twins that do not take the same parameters:\n  " + "\n  ".join(mismatched)


def test_everyAsyncAdminOperation_isACoroutine_andNoSyncOneIs() -> None:
    assert all(inspect.iscoroutinefunction(getattr(AsyncCyrockDbAdminClient, name))
               for name in _operations(AsyncCyrockDbAdminClient))
    assert not any(inspect.iscoroutinefunction(getattr(CyrockDbAdminClient, name))
                   for name in _operations(CyrockDbAdminClient))


def test_theAdminClients_haveAtLeastTheFirstScope() -> None:
    """A smoke check on the guard itself: an empty comparison would pass every test above."""
    expected = {"create_project", "list_projects", "delete_project", "create_api_key", "list_api_keys",
                "revoke_api_key", "close"}
    assert expected == set(_operations(CyrockDbAdminClient))
