"""The route key request header.

The counterpart of ``client-java``'s ``RouteKeyHeader`` and ``RouteKeyClientInterceptor``. Several data servers may
serve one storage, and each opens and holds the collections and graphs it is asked for. Every data-plane call carries
the store it is about as this header, and the gateway hashes on it, so that a store is opened on one data server
rather than on each. It is affinity, not ownership: any data server serves any call, with or without the header.

The gateway reads the header and never the request body, which would put a protobuf parser in front of every client
of its public port. Over REST it takes the store from the path instead.

A graph branch reads through its parents and keeps the root of that chain open, so it is routed by the root: a
branch's id starts with its root's id followed by :data:`BRANCH_SEPARATOR`, and the route key of any store id is the
part before the first separator, see :func:`route_key_for`. A branch created before this id form is routed by its id.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import grpc

from ._auth import Metadata, _ClientCallDetails

__all__ = [
    "DATA_PLANE_PREFIX",
    "MAX_LENGTH",
    "ROUTE_KEY_HEADER",
    "BRANCH_SEPARATOR",
    "STORE_FIELDS",
    "AsyncRouteKeyInterceptor",
    "RouteKeyInterceptor",
    "route_key_for",
    "route_key_of",
]

_TRequest  = TypeVar("_TRequest")
_TResponse = TypeVar("_TResponse")

ROUTE_KEY_HEADER = "x-cyrock-route-key"

# The request fields that name the store a call is about, in the order they are looked for, as in Java's
# RouteKeyHeader.STORE_FIELDS. A branch merge names the branch first: the branch is open where its own requests are
# routed, and its parent with it. tests/test_route_key.py holds every data-plane request to having one of them.
STORE_FIELDS = (
    "collection_id",
    "graph_id",
    "branch_graph_id",
    "resource_id",
)

# The method path prefix of the data-plane services, the calls the header is sent on.
DATA_PLANE_PREFIX = "/ai.cyrock.db.data.server."

# The longest store id sent as a route key; a longer one is left to the gateway's balancing.
MAX_LENGTH = 256

# Separates a branch id's root id from the branch's own part, as Java's RouteKeyHeader.BRANCH_SEPARATOR.
BRANCH_SEPARATOR = "~"


def route_key_for(store_id: str) -> str:
    """The route key of a store id: the id up to the first :data:`BRANCH_SEPARATOR`, the root of a branch.

    :param store_id: a collection or graph id
    :return: the part before the first separator, or the whole id if it holds none or starts with one
    """
    separator = store_id.find(BRANCH_SEPARATOR)
    return store_id[:separator] if separator > 0 else store_id


def route_key_of(request: Any) -> str | None:
    """The store a request names, as its route key.

    :param request: a protobuf request message
    :return: the :func:`route_key_for` of its first singular string field among :data:`STORE_FIELDS`, or ``None`` if
        it has none, the field is empty, or the key is not printable ASCII of at most :data:`MAX_LENGTH` characters,
        which is what a metadata value allows
    """
    descriptor = getattr(request, "DESCRIPTOR", None)
    if descriptor is None:
        return None
    for name in STORE_FIELDS:
        field = descriptor.fields_by_name.get(name)
        if field is None or field.type != field.TYPE_STRING or _is_repeated(field):
            continue
        value = route_key_for(getattr(request, name))
        return value if _is_valid_value(value) else None
    return None


def _is_repeated(field: Any) -> bool:
    # protobuf 5 deprecates FieldDescriptor.label in favour of is_repeated; both name the same thing.
    is_repeated = getattr(field, "is_repeated", None)
    if isinstance(is_repeated, bool):
        return is_repeated
    return bool(field.label == field.LABEL_REPEATED)


def _is_valid_value(value: str) -> bool:
    return 0 < len(value) <= MAX_LENGTH and all(0x20 <= ord(character) <= 0x7E for character in value)


def _method_of(details: Any) -> str:
    # grpc.aio reports the method as bytes, the synchronous interceptor as str; see _auth.AsyncAuthInterceptor.
    method = details.method
    return method.decode() if isinstance(method, bytes) else str(method)


def _route_key_for(details: Any, request: Any) -> str | None:
    """The route key to add to a call, or ``None`` if it is no data-plane call, names no store, or has one already."""
    if not _method_of(details).startswith(DATA_PLANE_PREFIX):
        return None
    if any(key == ROUTE_KEY_HEADER for key, _value in (details.metadata or ())):
        return None
    return route_key_of(request)


class RouteKeyInterceptor(
    grpc.UnaryUnaryClientInterceptor,
    grpc.UnaryStreamClientInterceptor,
):
    """Adds the route key header to every data-plane call, unary and server streaming.

    A header the caller set already is kept. Calls outside the data plane pass through untouched.
    """

    @staticmethod
    def _with_route_key(details: grpc.ClientCallDetails, request: Any) -> grpc.ClientCallDetails:
        route_key = _route_key_for(details, request)
        if route_key is None:
            return details
        metadata: list[tuple[str, str | bytes]] = list(details.metadata) if details.metadata else []
        metadata.append((ROUTE_KEY_HEADER, route_key))
        extended: Metadata = metadata
        return _ClientCallDetails(
            method         = details.method,
            timeout        = details.timeout,
            metadata       = extended,
            credentials    = details.credentials,
            wait_for_ready = getattr(details, "wait_for_ready", None),
            compression    = getattr(details, "compression", None),
        )

    def intercept_unary_unary(
        self,
        continuation:        Callable[[grpc.ClientCallDetails, _TRequest], _TResponse],
        client_call_details: grpc.ClientCallDetails,
        request:             _TRequest,
    ) -> _TResponse:
        return continuation(self._with_route_key(client_call_details, request), request)

    def intercept_unary_stream(
        self,
        continuation:        Callable[[grpc.ClientCallDetails, _TRequest], grpc._CallIterator[_TResponse]],
        client_call_details: grpc.ClientCallDetails,
        request:             _TRequest,
    ) -> grpc._CallIterator[_TResponse]:
        return continuation(self._with_route_key(client_call_details, request), request)


class AsyncRouteKeyInterceptor(
    grpc.aio.UnaryUnaryClientInterceptor,
    grpc.aio.UnaryStreamClientInterceptor,
):
    """The asyncio counterpart of :class:`RouteKeyInterceptor`; pass it to a channel as :meth:`interceptors`."""

    @staticmethod
    def _with_route_key(details: Any, request: Any) -> Any:
        route_key = _route_key_for(details, request)
        if route_key is None:
            return details
        metadata = grpc.aio.Metadata(*(details.metadata or ()))
        metadata[ROUTE_KEY_HEADER] = route_key
        return details._replace(metadata=metadata)

    async def intercept_unary_unary(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        return await continuation(self._with_route_key(client_call_details, request), request)

    async def intercept_unary_stream(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        return await continuation(self._with_route_key(client_call_details, request), request)

    def interceptors(self) -> list[grpc.aio.ClientInterceptor]:
        """This interceptor in the form a grpc.aio channel takes it: once for unary calls, once for streams.

        grpc.aio files each interceptor under the first call type it implements and never consults it for the
        others, as for :meth:`~cyrock_db._auth.AsyncAuthInterceptor.interceptors`.
        """
        return [self, _AsyncStreamView(self)]


class _AsyncStreamView(grpc.aio.UnaryStreamClientInterceptor):
    """An :class:`AsyncRouteKeyInterceptor` as grpc.aio sees a stream interceptor."""

    def __init__(self, owner: AsyncRouteKeyInterceptor) -> None:
        self._owner = owner

    async def intercept_unary_stream(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        return await self._owner.intercept_unary_stream(continuation, client_call_details, request)
