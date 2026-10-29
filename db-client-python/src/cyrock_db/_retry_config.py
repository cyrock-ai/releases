"""Turns a :class:`~cyrock_db.types.RetryPolicy` into the gRPC service config that makes the channel retry.

Mirrors ``RetryServiceConfig`` in client-java. Which operations a policy covers is read from the protocol,
not listed here: a read is marked ``option idempotency_level = NO_SIDE_EFFECTS`` in the ``.proto`` files, and
the generated descriptors carry that option. Java reads the same option, so both SDKs retry the same
operations. Only unary methods qualify: the one streaming call, Change Data Capture, reconnects on its own and
must not be replayed by the channel.

The config is gRPC's own, described in gRFC A6 (client retries).
"""

from __future__ import annotations

import json
from typing import Any

from google.protobuf import descriptor_pb2
from google.protobuf.descriptor import MethodDescriptor, ServiceDescriptor

from ._proto import cyrock_db_data_pb2 as data_pb2
from ._proto import cyrock_db_graph_pb2 as graph_pb2
from ._proto import cyrock_db_nlsearch_pb2 as nlsearch_pb2
from ._proto import cyrock_db_platform_pb2 as platform_pb2
from .types import RetryPolicy

__all__ = ["retried_methods", "service_config"]

# Every service this client calls, the token exchange included.
_SERVICES: tuple[ServiceDescriptor, ...] = (
    data_pb2.DESCRIPTOR.services_by_name["DocumentGrpcService"],
    graph_pb2.DESCRIPTOR.services_by_name["GraphGrpcService"],
    nlsearch_pb2.DESCRIPTOR.services_by_name["NlSearchGrpcService"],
    platform_pb2.DESCRIPTOR.services_by_name["CollectionDefinitionGrpcService"],
    platform_pb2.DESCRIPTOR.services_by_name["GraphDefinitionGrpcService"],
    platform_pb2.DESCRIPTOR.services_by_name["TokenGrpcService"],
)

_REPEATABLE = (descriptor_pb2.MethodOptions.NO_SIDE_EFFECTS, descriptor_pb2.MethodOptions.IDEMPOTENT)


def _retried(method: MethodDescriptor, policy: RetryPolicy) -> bool:
    if method.client_streaming or method.server_streaming:
        return False
    return policy.retry_writes or method.GetOptions().idempotency_level in _REPEATABLE


def retried_methods(policy: RetryPolicy) -> list[str]:
    """The full names (``package.Service/Method``) of the methods ``policy`` retries.

    The unary methods of the services this client calls that are marked free of side effects or
    idempotent, or all of them when the policy retries writes; in service and declaration order.
    """
    return [
        f"{service.full_name}/{method.name}"
        for service in _SERVICES
        for method in service.methods
        if _retried(method, policy)
    ]


def _seconds(value: float) -> str:
    """A duration as the service config writes one: seconds with nine fractional digits and an ``s``."""
    nanos = round(value * 1_000_000_000)
    return f"{nanos // 1_000_000_000}.{nanos % 1_000_000_000:09d}s"


def service_config(policy: RetryPolicy) -> str:
    """The service config for ``policy``, as the JSON the ``grpc.service_config`` channel argument takes."""
    names = [
        {"service": name.split("/")[0], "method": name.split("/")[1]}
        for name in retried_methods(policy)
    ]
    retry_policy: dict[str, Any] = {
        "maxAttempts":          policy.max_attempts,
        "initialBackoff":       _seconds(policy.initial_backoff),
        "maxBackoff":           _seconds(policy.max_backoff),
        "backoffMultiplier":    policy.backoff_multiplier,
        "retryableStatusCodes": ["UNAVAILABLE"],
    }
    return json.dumps({"methodConfig": [{"name": names, "retryPolicy": retry_policy}]})
