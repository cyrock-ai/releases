"""Change stream subscriptions.

A port of ``ChangeStreamSubscription``. The interesting behaviour is not the call - it is one
server-streaming RPC - but what happens around it, and every rule below is one the Java version
arrived at for a reason worth keeping.

**Reconnection.** A transient failure reconnects with exponential backoff, 1s doubling to 60s,
resuming from the last delivered LSN. The backoff resets on any received event, so a stream that
flaps does not inherit a minute-long delay from an outage an hour ago. Because a fresh token is
fetched on each connect, this also survives token expiry over a long-lived stream - which is why
``UNAUTHENTICATED`` is treated as retryable rather than fatal.

**Permanent failures stop.** ``PERMISSION_DENIED``, ``FAILED_PRECONDITION``, ``INVALID_ARGUMENT``,
``UNIMPLEMENTED`` and ``NOT_FOUND`` are not worth retrying: they will fail identically forever.
``FAILED_PRECONDITION`` is the one to expect in practice - watching a branch returns it, branches
being overlays with no change log of their own.

**The cursor advances past a discontinuity, deliberately.** A ``DISCONTINUITY`` marker is a real
record at a real LSN, and not advancing past it would have every reconnect receive it again forever.
What is lost is lost: resuming from before it cannot bring the changes back. The consumer is told,
and must re-snapshot.

**Snapshot events do not advance the cursor**, because they are not log positions; the
``snapshot_complete`` marker does, because it is.

**A listener that raises does not stop the stream.** The exception is logged with its traceback and the
subscription reconnects after the usual backoff, resuming after the event whose callback raised - the cursor
already stood at it - so that event is not delivered again. This is what the Java subscription does: gRPC
cancels the call whose ``onNext`` threw, and the cancellation is a transient failure like any other. Only a
permanent failure reaches ``on_error``.

**The live marker tells the listener the stream is live**, through ``on_live``, after the first connect
and after every reconnect. A subscription connects in the background, so a caller that watches from
now and then writes must wait for it first, or its first writes may precede the stream's start. The
marker's LSN is the stream's position, so the cursor advances to it and a reconnect resumes after it
rather than from now.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Protocol

import grpc

from . import _proto_converters as converters
from ._proto import cyrock_db_cdc_pb2 as cdc_pb2
from ._proto import cyrock_db_cdc_pb2_grpc as cdc_grpc
from ._values_codec import to_struct
from .exceptions import from_rpc_error
from .types import ChangeEvent, ChangeOp, WatchOptions, WatchStart

__all__ = ["ChangeListener", "ChangeStreamSubscription", "watch_request"]

LOG = logging.getLogger(__name__)

INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS     = 60.0

# Failures that will fail the same way on every retry. UNAUTHENTICATED is deliberately absent: a
# reconnect fetches a fresh token, so an expired one recovers by itself.
PERMANENT_FAILURES = frozenset({
    grpc.StatusCode.PERMISSION_DENIED,
    grpc.StatusCode.FAILED_PRECONDITION,
    grpc.StatusCode.INVALID_ARGUMENT,
    grpc.StatusCode.UNIMPLEMENTED,
    grpc.StatusCode.NOT_FOUND,
})


class ChangeListener(Protocol):
    """What a subscriber implements. Only :meth:`on_change` is required.

    A plain callable taking one :class:`~cyrock_db.types.ChangeEvent` is accepted anywhere a listener
    is, which covers the common case without a class.

    A listener may also define any of these hooks, which the subscription calls when present:

    - ``on_discontinuity(self, event: ChangeEvent) -> None``: called before :meth:`on_change` when
      changes have been lost.
    - ``on_snapshot_complete(self) -> None``: called once, after the initial snapshot and before the
      live tail.
    - ``on_live(self) -> None``: called each time the stream is live, after the first connect and
      every reconnect.
    - ``on_error(self, error: BaseException) -> None``: called once when the stream stops permanently.

    They are deliberately not members of this protocol. A protocol member is required of every
    implementation, so listing them made a type checker reject a listener with only
    :meth:`on_change`, and each hook added later would have rejected every listener written before
    it. Java has the same hooks as default methods, which a Python protocol cannot express.
    """

    def on_change(self, event: ChangeEvent) -> None:
        """Called for every event, including a discontinuity marker."""


def watch_request(
    resource_kind: Any,
    resource_id:   str,
    options:       WatchOptions,
    from_lsn:      int,
    live_marker:   bool = False,
) -> Any:
    """Builds a ``WatchRequest``.

    ``from_lsn`` is passed separately from ``options`` because a reconnect resumes from the last
    delivered LSN rather than from wherever the caller originally asked to start - the options say
    where the *subscription* began, not where this particular connection should. ``live_marker`` asks
    for the ``live`` marker event, which only the listener-based subscription handles; the async
    iterator leaves it off so its callers see no event they did not see before.
    """
    request = cdc_pb2.WatchRequest(
        resource_kind   = resource_kind,
        resource_id     = resource_id,
        include_vectors = options.include_vectors,
        live_marker     = live_marker,
    )
    if from_lsn >= 0:
        request.from_lsn = from_lsn
    elif options.start is WatchStart.WITH_SNAPSHOT:
        request.with_snapshot = True
    elif options.start is WatchStart.FROM_LSN:
        request.from_lsn = options.from_lsn
    else:
        request.from_now = True

    change_filter = options.filter
    if change_filter.ops:
        request.filter.ops.extend(cdc_pb2.ChangeOp.Value(each.value) for each in change_filter.ops)
    if change_filter.entity_kinds:
        request.filter.entity_kinds.extend(
            cdc_pb2.EntityKind.Value(each.value) for each in change_filter.entity_kinds
        )
    if change_filter.labels:
        request.filter.labels.extend(change_filter.labels)
    if change_filter.metadata_predicate:
        request.filter.metadata_predicate = change_filter.metadata_predicate
        if change_filter.metadata_predicate_parameters:
            request.filter.metadata_predicate_parameters.CopyFrom(
                to_struct(change_filter.metadata_predicate_parameters)
            )
    return request


class ChangeStreamSubscription:
    """A running change stream. Close it to stop, or use it as a context manager.

    The stream runs on its own daemon thread, so a caller that forgets to close one does not keep the
    process alive. Callbacks run on that thread: keep them short, or hand off to your own queue.

    A callback that raises does not stop the stream: the exception is logged, and the subscription reconnects
    after a backoff and resumes after the event whose callback raised, which is not delivered again. A
    snapshot row is the exception, since snapshot rows do not move the cursor: the snapshot is sent again.
    Handle what must not be skipped inside the callback.

    Created by :meth:`CyrockDbClient.watch_graph <cyrock_db.CyrockDbClient.watch_graph>` and
    :meth:`~cyrock_db.CyrockDbClient.watch_collection`, not directly.
    """

    def __init__(
        self,
        stub:          Any,
        resource_kind: Any,
        resource_id:   str,
        options:       WatchOptions,
        listener:      ChangeListener | Callable[[ChangeEvent], None],
    ) -> None:
        self._stub          = stub
        self._resource_kind = resource_kind
        self._resource_id   = resource_id
        self._options       = options
        self._listener      = listener
        self._closed        = threading.Event()
        self._last_lsn      = -1
        self._backoff       = INITIAL_BACKOFF_SECONDS
        self._call: Any     = None
        self._thread        = threading.Thread(
            target=self._run, name=f"cdc-watch-{resource_id}", daemon=True
        )
        self._thread.start()

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    @property
    def last_lsn(self) -> int:
        """The last LSN delivered, the value to checkpoint, or -1 before the first event.

        -1 names no log position and ``WatchOptions.resume_from`` refuses it: a consumer with nothing applied yet
        opens its stream with ``WatchOptions.with_snapshot()`` or ``resume_from(0)``.
        """
        return self._last_lsn

    @property
    def closed(self) -> bool:
        """Whether the stream has stopped: closed by the caller, or ended by a permanent failure."""
        return self._closed.is_set()

    def close(self) -> None:
        """Stops the stream. Idempotent, and safe to call from a listener callback."""
        self._closed.set()
        self._cancel_call()

    def __enter__(self) -> ChangeStreamSubscription:
        return self

    def __exit__(
        self,
        exc_type:  type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ─── Internals ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                self._consume()
            except grpc.RpcError as error:
                if self._closed.is_set():
                    return
                code = error.code() if callable(getattr(error, "code", None)) else None
                if code in PERMANENT_FAILURES:
                    LOG.warning(
                        "Change stream for %s failed permanently: %s", self._resource_id, error
                    )
                    self._closed.set()
                    self._notify_error(from_rpc_error(error))
                    return
                LOG.warning(
                    "Change stream for %s failed, reconnecting: %s", self._resource_id, error
                )
            except _ListenerFailed as failed:
                if self._closed.is_set():
                    return
                # As in Java, where gRPC cancels the call whose onNext threw and the subscription treats the
                # cancellation as a transient failure: drop this connection, back off, and reconnect from the
                # cursor. The cursor already stands at the event whose callback raised, so that event is not
                # delivered again.
                LOG.error(
                    "Change stream listener for %s raised; reconnecting after LSN %s",
                    self._resource_id, self._last_lsn, exc_info=failed.__cause__,
                )
                self._cancel_call()
            except Exception as error:  # noqa: BLE001 - must end the stream, not kill the thread silently
                # Not the listener's: decoding the event or the subscription's own bookkeeping failed. That
                # fails the same way on every reconnect, so the stream stops and the caller is told.
                LOG.exception("Change stream for %s failed", self._resource_id)
                if self._closed.is_set():
                    return
                self._closed.set()
                self._cancel_call()
                self._notify_error(error)
                return

            # The server ending the stream cleanly is treated as a disconnect too: the caller asked
            # to keep watching, so reconnect rather than stop silently.
            if self._closed.wait(self._next_backoff()):
                return

    def _cancel_call(self) -> None:
        call = self._call
        if call is not None:
            call.cancel()

    def _next_backoff(self) -> float:
        delay = self._backoff
        self._backoff = min(self._backoff * 2, MAX_BACKOFF_SECONDS)
        return delay

    def _consume(self) -> None:
        request   = watch_request(
            self._resource_kind, self._resource_id, self._options, self._last_lsn, live_marker=True
        )
        self._call = self._stub.WatchChanges(request)
        for message in self._call:
            if self._closed.is_set():
                return
            self._deliver(converters.to_change_event(message))
            # A delivered event means the connection works, so a later outage starts its backoff from the
            # beginning rather than inheriting a delay from an unrelated one. Only once the listener took it: a
            # snapshot row does not move the cursor, so one the listener keeps failing on comes again on every
            # reconnect, which must then back off up to the cap rather than retry every second.
            self._backoff = INITIAL_BACKOFF_SECONDS

    def _deliver(self, event: ChangeEvent) -> None:
        if event.snapshot_complete:
            # A real record at a real LSN, so the cursor advances even though it is a marker.
            self._last_lsn = max(self._last_lsn, event.source.lsn)
            self._call_listener("on_snapshot_complete")
            return

        if event.live:
            # The stream's position, so a reconnect resumes after it rather than from now.
            self._last_lsn = max(self._last_lsn, event.source.lsn)
            self._call_listener("on_live")
            return

        if event.op is not ChangeOp.SNAPSHOT:
            # Snapshot rows are current state, not log positions, so they do not move the cursor.
            self._last_lsn = event.source.lsn

        if event.op is ChangeOp.DISCONTINUITY:
            LOG.warning(
                "Change stream for %s has a discontinuity at LSN %s: changes before it were committed "
                "but never logged, so this consumer's state is stale until it re-snapshots",
                self._resource_id, event.source.lsn,
            )
            self._call_listener("on_discontinuity", event)

        self._call_listener("on_change", event)

    def _invoke_listener(self, name: str, *args: Any) -> None:
        """Invokes an optional listener method, tolerating a plain callable and missing hooks."""
        if name == "on_change" and callable(self._listener) and not hasattr(self._listener, "on_change"):
            self._listener(*args)
            return
        hook = getattr(self._listener, name, None)
        if hook is not None:
            hook(*args)

    def _call_listener(self, name: str, *args: Any) -> None:
        """Invokes a listener hook, marking what it raises as the listener's, see :class:`_ListenerFailed`."""
        try:
            self._invoke_listener(name, *args)
        except Exception as error:  # noqa: BLE001 - re-raised, marked as the listener's
            raise _ListenerFailed from error

    def _notify_error(self, error: BaseException) -> None:
        hook = getattr(self._listener, "on_error", None)
        if hook is not None:
            hook(error)


class _ListenerFailed(Exception):
    """A listener hook raised; the hook's exception is the ``__cause__``.

    Kept apart from every other exception of the consuming loop because only a listener's failure is one a
    reconnect gets past: decoding an event or the subscription's own bookkeeping would fail again on every
    reconnect, so those end the stream and reach ``on_error`` instead.
    """


def iter_changes(
    stub: Any, resource_kind: Any, resource_id: str, options: WatchOptions
) -> Iterator[ChangeEvent]:
    """A single-pass iterator over a change stream, with no reconnection.

    Used by the asynchronous client's iterator form, and available where a caller wants the raw
    stream. The reconnect, backoff and cursor behaviour lives in
    :class:`ChangeStreamSubscription`; this is deliberately the plain version.
    """
    for message in stub.WatchChanges(watch_request(resource_kind, resource_id, options, -1)):
        yield converters.to_change_event(message)


CDC_STUB = cdc_grpc.ChangeStreamGrpcServiceStub
