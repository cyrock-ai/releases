"""Attaches credentials to outgoing calls.

**Why interceptors and not call credentials.** ``client-java`` uses ``CallCredentials``
(``ApiKeyCallCredentials``, ``JwtCallCredentials``), which grpc-java applies over plaintext as
readily as over TLS. grpc-python refuses: a call credential on an insecure channel fails the call
with ``UNAUTHENTICATED`` and "Established channel does not have a sufficient security level to
transfer call credential". The client's default is plaintext against a local gateway, so call
credentials would break the common case outright. A client interceptor adds the same metadata and
does not care about the channel's security level.

The headers are the same either way, and are the ones the server reads: ``x-api-key`` for an API
key, ``authorization: Bearer <jwt>`` for an exchanged token.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

import grpc

# grpc._CallIterator is declared by the type stubs and does not exist at runtime. `from __future__
# import annotations` keeps every annotation a string, so it is never evaluated - do not "fix" this
# by importing it.
_TRequest  = TypeVar("_TRequest")
_TResponse = TypeVar("_TResponse")

# gRPC allows binary metadata values on keys ending in "-bin". None of ours are binary, but a caller
# may have set one on the call before this interceptor rebuilds the details, so the type has to hold it.
Metadata = Sequence[tuple[str, str | bytes]]

__all__ = [
    "API_KEY_HEADER",
    "API_KEY_SERVICES",
    "AUTHORIZATION_HEADER",
    "TOKEN_EXCHANGE_METHOD",
    "AsyncAuthInterceptor",
    "AuthInterceptor",
    "api_key_header",
    "bearer",
    "bearer_token",
    "uses_api_key",
]


def api_key_header(api_key: str) -> tuple[str, str]:
    """The ``x-api-key`` header."""
    return (API_KEY_HEADER, api_key)

API_KEY_HEADER       = "x-api-key"
AUTHORIZATION_HEADER = "authorization"

# The exchange that produces the token must not carry the token it is about to produce. The
# synchronous client keeps the token stub on the un-intercepted channel, which grpc-python allows
# because grpc.intercept_channel wraps an existing channel. grpc.aio takes its interceptors at
# construction and offers no such wrapper, so the asynchronous client uses one channel and has the
# interceptor step aside for this method instead.
TOKEN_EXCHANGE_METHOD = "/ai.cyrock.db.platform.server.TokenGrpcService/Exchange"

# Services that authenticate with the raw API key rather than with the exchanged JWT.
#
# **This is not a plane split, and reading it as one is how it goes wrong.** The exchanged token is a
# data-plane credential: the platform server issues it and then refuses it on its own definition
# services, which answer UNAUTHENTICATED "Invalid bearer token". So the split is per service, exactly
# as client-java wires its stubs - definition services get ApiKeyCallCredentials, documents, graphs,
# NL search and CDC get JwtCallCredentials.
#
# The list is kept here because the client needs it at runtime and a published wheel has no reactor
# beside it to read the contract from. But it is no longer the only copy, which is what #464 fixed:
# `proto/src/main/resources/credential-by-service.tsv` is now the one authority, the Java build holds
# the client wiring to it, and `tests/test_auth.py` holds this tuple to it - so a drift here fails a
# build rather than surfacing as a runtime UNAUTHENTICATED against a real server. Keep the two in step;
# do not edit this without editing the table.
API_KEY_SERVICES = (
    "/ai.cyrock.db.platform.server.CollectionDefinitionGrpcService/",
    "/ai.cyrock.db.platform.server.GraphDefinitionGrpcService/",
    "/ai.cyrock.db.platform.server.TokenGrpcService/",
)


def uses_api_key(method: str) -> bool:
    """Whether this method authenticates with the API key rather than the exchanged token."""
    return method.startswith(API_KEY_SERVICES)


def bearer(token: str) -> tuple[str, str]:
    """The ``authorization`` header for a JWT."""
    return (AUTHORIZATION_HEADER, f"Bearer {token}")


def bearer_token(metadata: Metadata) -> str | None:
    """The JWT in an ``authorization`` header among ``metadata``, or ``None`` if there is none."""
    prefix = "Bearer "
    for key, value in metadata:
        if key == AUTHORIZATION_HEADER and isinstance(value, str) and value.startswith(prefix):
            return value[len(prefix):]
    return None


class _ClientCallDetails(grpc.ClientCallDetails):
    """A rebuilt call description.

    ``grpc.ClientCallDetails`` carries no setters and the concrete type gRPC passes in is an
    immutable named tuple, so adding a header means constructing a replacement rather than editing
    the original.
    """

    def __init__(
        self,
        method:         str,
        timeout:        float | None,
        metadata:       Metadata | None,
        credentials:    grpc.CallCredentials | None,
        wait_for_ready: bool | None,
        compression:    Any,
    ) -> None:
        self.method         = method
        self.timeout        = timeout
        self.metadata       = tuple(metadata) if metadata is not None else None
        self.credentials    = credentials
        self.wait_for_ready = wait_for_ready
        self.compression    = compression


def remaining(timeout: float | None, started: float) -> float | None:
    """What is left of a call's ``timeout`` after the time since ``started`` went to obtaining its credential.

    The interceptors fetch the token before the call starts, where Java's call credentials do it inside a
    call whose deadline is already running. Taking the wait off the call keeps ``call_timeout`` the budget
    for the whole call, token exchange and retries included, as it is in Java. A budget already spent leaves
    zero, which gRPC answers with ``DEADLINE_EXCEEDED`` rather than sending the request.
    """
    if timeout is None:
        return None
    return max(timeout - (time.monotonic() - started), 0.0)


class AuthInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
):
    """Adds credential metadata to every call.

    Both the unary and the server-streaming form are intercepted: a Change Data Capture subscription
    is a streaming call and needs authorizing exactly like any other. The client-streaming forms are
    not implemented because the schema has none.

    The provider is given the call's method path and produces the metadata for it, because the
    credential is not the same for every service - see :data:`API_KEY_SERVICES`. Producing it per call
    also means a token that expires mid-life is refreshed by the provider rather than pinned here.

    A call the server answers with ``UNAUTHENTICATED`` is reported to ``on_rejected`` with the credential
    it carried, so the provider's owner can drop a token the server no longer accepts instead of reusing
    it until it expires. The call itself still fails: whether it is safe to repeat is the caller's call.
    """

    def __init__(
        self,
        metadata_provider: Callable[[str], Sequence[tuple[str, str]]],
        skip_methods:      frozenset[str] = frozenset(),
        on_rejected:       Callable[[Metadata], None] | None = None,
    ) -> None:
        self._metadata_provider = metadata_provider
        self._skip_methods      = skip_methods
        self._on_rejected       = on_rejected

    def _with_auth(self, details: grpc.ClientCallDetails) -> tuple[_ClientCallDetails, Metadata]:
        existing: list[tuple[str, str | bytes]] = list(details.metadata) if details.metadata else []
        started    = time.monotonic()
        credential = list(self._metadata_provider(str(details.method)))
        existing.extend(credential)
        return _ClientCallDetails(
            method         = details.method,
            timeout        = remaining(details.timeout, started),
            metadata       = existing,
            credentials    = details.credentials,
            wait_for_ready = getattr(details, "wait_for_ready", None),
            compression    = getattr(details, "compression", None),
        ), credential

    def _watch(self, call: Any, credential: Metadata) -> None:
        """Reports the credential to ``on_rejected`` once the call ends ``UNAUTHENTICATED``.

        Both outcomes a continuation returns - a call still running and one that already failed - are
        futures, and a done callback on one that already finished runs at once.
        """
        on_rejected = self._on_rejected
        if on_rejected is None:
            return

        def report(done: Any) -> None:
            if done.code() == grpc.StatusCode.UNAUTHENTICATED:
                on_rejected(credential)

        call.add_done_callback(report)

    def intercept_unary_unary(
        self,
        continuation:        Callable[[grpc.ClientCallDetails, _TRequest], _TResponse],
        client_call_details: grpc.ClientCallDetails,
        request:             _TRequest,
    ) -> _TResponse:
        if client_call_details.method in self._skip_methods:
            return continuation(client_call_details, request)
        details, credential = self._with_auth(client_call_details)
        call = continuation(details, request)
        self._watch(call, credential)
        return call

    def intercept_unary_stream(
        self,
        continuation:        Callable[[grpc.ClientCallDetails, _TRequest], grpc._CallIterator[_TResponse]],
        client_call_details: grpc.ClientCallDetails,
        request:             _TRequest,
    ) -> grpc._CallIterator[_TResponse]:
        if client_call_details.method in self._skip_methods:
            return continuation(client_call_details, request)
        details, credential = self._with_auth(client_call_details)
        call = continuation(details, request)
        self._watch(call, credential)
        return call


class AsyncAuthInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor,
    grpc.aio.UnaryStreamClientInterceptor,
):
    """The asyncio counterpart of :class:`AuthInterceptor`.

    grpc.aio interceptors are a separate hierarchy with async methods, so this cannot simply reuse
    the synchronous one. The provider is awaited, because obtaining a token may itself require a
    round trip. ``on_rejected`` is as for :class:`AuthInterceptor`.
    """

    def __init__(
        self,
        metadata_provider: Callable[[str], Awaitable[Sequence[tuple[str, str]]]],
        skip_methods:      frozenset[str] = frozenset(),
        on_rejected:       Callable[[Metadata], None] | None = None,
    ) -> None:
        self._metadata_provider = metadata_provider
        self._skip_methods      = skip_methods
        self._on_rejected       = on_rejected
        # The stream watchers are tasks nothing else refers to, and the event loop holds tasks only weakly.
        # Kept here until they finish, or one could be collected while it still waits on its call.
        self._watchers: set[asyncio.Task[None]] = set()

    @staticmethod
    def _method_of(details: Any) -> str:
        """The call's method path, as text.

        grpc.aio reports ``method`` as **bytes**, where the synchronous interceptor reports ``str``.
        Comparing the raw value against a str skip-list therefore never matches - and the failure is
        not a missed header but a hang: the token exchange goes through the interceptor, which awaits
        the token that exchange is in the middle of producing, and the two wait on each other for
        ever. Normalising here is what keeps that impossible.
        """
        method = details.method
        return method.decode() if isinstance(method, bytes) else str(method)

    async def _with_auth(self, details: Any) -> tuple[Any, Metadata]:
        metadata   = grpc.aio.Metadata(*(details.metadata or ()))
        started    = time.monotonic()
        credential = list(await self._metadata_provider(self._method_of(details)))
        for key, value in credential:
            metadata[key] = value
        return details._replace(metadata=metadata, timeout=remaining(details.timeout, started)), credential

    async def _report(self, call: Any, credential: Metadata) -> None:
        """Waits for the call's status, and reports the credential to ``on_rejected`` if it is ``UNAUTHENTICATED``.

        Waiting on the status does not consume the responses, which stay with the caller.
        """
        if self._on_rejected is not None and await call.code() == grpc.StatusCode.UNAUTHENTICATED:
            self._on_rejected(credential)

    async def intercept_unary_unary(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        if self._method_of(client_call_details) in self._skip_methods:
            return await continuation(client_call_details, request)
        details, credential = await self._with_auth(client_call_details)
        call = await continuation(details, request)
        if self._on_rejected is not None:
            # Waited for here rather than in the background, so the token is dropped before the caller sees
            # the failure and its next call cannot race ahead with the rejected token. A unary caller waits
            # for the whole call anyway. Cancelling the caller cancels this wait, not the call it waits on,
            # so the call is cancelled with it.
            try:
                await self._report(call, credential)
            except asyncio.CancelledError:
                call.cancel()
                raise
        return call

    def interceptors(self) -> list[grpc.aio.ClientInterceptor]:
        """This interceptor in the form a grpc.aio channel takes it: once for unary calls, once for streams.

        grpc.aio files each interceptor under the first call type it implements and never consults it for
        the others, so one object implementing both sees only unary calls. Passed on its own, the change
        stream went out with no credential at all. The second entry is a stream-only view of this same
        interceptor, so both call types share one provider and one set of watchers.
        """
        return [self, _AsyncStreamView(self)]

    async def intercept_unary_stream(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        if self._method_of(client_call_details) in self._skip_methods:
            return await continuation(client_call_details, request)
        details, credential = await self._with_auth(client_call_details)
        call = await continuation(details, request)
        if self._on_rejected is not None:
            # A stream stays open for as long as the caller reads it, so its status is waited for in the
            # background. The event loop holds tasks only weakly, hence the set.
            watcher = asyncio.get_running_loop().create_task(self._report(call, credential))
            self._watchers.add(watcher)
            watcher.add_done_callback(self._watchers.discard)
        return call


class _AsyncStreamView(grpc.aio.UnaryStreamClientInterceptor):
    """An :class:`AsyncAuthInterceptor` as grpc.aio sees a stream interceptor.

    See :meth:`AsyncAuthInterceptor.interceptors`.
    """

    def __init__(self, owner: AsyncAuthInterceptor) -> None:
        self._owner = owner

    async def intercept_unary_stream(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        return await self._owner.intercept_unary_stream(continuation, client_call_details, request)
