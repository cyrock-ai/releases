"""The listener types the blocking client's ``watch_graph`` and ``watch_collection`` accept.

``ChangeListener`` is a protocol, so whether a listener fits is decided by a type checker, not at
runtime. This runs mypy, as strict as the package's own check, on listeners a caller may write, so
that the protocol cannot again require hooks the documentation calls optional.
"""

from __future__ import annotations

from pathlib import Path

from mypy import api

PROJECT = Path(__file__).resolve().parents[1]

LISTENERS = '''
from cyrock_db import CyrockDbClient
from cyrock_db.types import ChangeEvent, WatchOptions


class OnlyOnChange:
    def on_change(self, event: ChangeEvent) -> None:
        pass


class SomeHooks:
    def on_change(self, event: ChangeEvent) -> None:
        pass

    def on_snapshot_complete(self) -> None:
        pass

    def on_error(self, error: BaseException) -> None:
        pass


def on_event(event: ChangeEvent) -> None:
    pass


def watch(client: CyrockDbClient) -> None:
    client.watch_graph("graph", WatchOptions.from_now(), OnlyOnChange()).close()
    client.watch_collection("collection", WatchOptions.from_now(), SomeHooks()).close()
    client.watch_graph("graph", WatchOptions.from_now(), on_event).close()
'''

NOT_A_LISTENER = '''
from cyrock_db import CyrockDbClient
from cyrock_db.types import WatchOptions


class NoOnChange:
    def on_live(self) -> None:
        pass


def watch(client: CyrockDbClient) -> None:
    client.watch_graph("graph", WatchOptions.from_now(), NoOnChange()).close()
'''


def _mypy(
    source:   str,
    tmp_path: Path,
) -> tuple[str, int]:
    module = tmp_path / "listeners.py"
    module.write_text(source)
    report, _, status = api.run([
        "--strict",
        "--config-file", str(PROJECT / "pyproject.toml"),
        "--cache-dir", str(PROJECT / ".mypy_cache"),
        str(module),
    ])
    return report, status


def test_watch_listenerWithOnChangeAndAnySubsetOfHooks_typeChecks(tmp_path: Path) -> None:
    report, status = _mypy(LISTENERS, tmp_path)

    assert status == 0, report


def test_watch_listenerWithoutOnChange_isRejected(tmp_path: Path) -> None:
    report, status = _mypy(NOT_A_LISTENER, tmp_path)

    assert status == 1, report
    assert 'Argument 3 to "watch_graph"' in report, report
