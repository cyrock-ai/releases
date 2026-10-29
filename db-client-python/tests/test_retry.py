"""Pins which calls a :class:`RetryPolicy` covers and that both clients really retry them.

The counterpart of ``RetryPolicyTest`` and ``RetryServiceConfigTest`` in client-java. The coverage comes from
the ``idempotency_level`` options in the ``.proto`` files and the behaviour from gRPC, so both are checked
against the generated descriptors and a real server rather than restated.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any

import grpc
import pytest

from cyrock_db import UnavailableException
from cyrock_db._channel import ClientOptions
from cyrock_db._proto import cyrock_db_data_pb2 as data_pb2
from cyrock_db._proto import cyrock_db_data_pb2_grpc as data_grpc
from cyrock_db._proto import cyrock_db_platform_pb2 as platform_pb2
from cyrock_db._proto import cyrock_db_platform_pb2_grpc as platform_grpc
from cyrock_db._retry_config import retried_methods, service_config
from cyrock_db.aio import AsyncCyrockDbClient
from cyrock_db.client import CyrockDbClient
from cyrock_db.types import RetryPolicy

# The shortest backoffs a policy accepts, so a test retrying four times stays fast.
FAST = RetryPolicy(initial_backoff=0.01, max_backoff=0.02)

DOCUMENTS = "ai.cyrock.db.data.server.DocumentGrpcService"
GRAPH     = "ai.cyrock.db.data.server.GraphGrpcService"
TOKENS    = "ai.cyrock.db.platform.server.TokenGrpcService"


def test_RetryPolicy_untouched_givesTheDocumentedDefaults() -> None:
    policy = RetryPolicy()

    assert policy.max_attempts       == 5
    assert policy.initial_backoff    == 0.5
    assert policy.max_backoff        == 10.0
    assert policy.backoff_multiplier == 2.0
    assert policy.retry_writes is False


def test_RetryPolicy_maxAttemptsOutsideTwoToFive_isRejected() -> None:
    with pytest.raises(ValueError, match="max_attempts must be between 2 and 5, was 1"):
        RetryPolicy(max_attempts=1)
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=6)


def test_RetryPolicy_tooShortOrNonFiniteBackoff_isRejected() -> None:
    # Below 10 ms grpc-python can round a retry delay to zero, and a zero delay never fires.
    with pytest.raises(ValueError, match="initial_backoff must be .* at least 0.01, was 0.001"):
        RetryPolicy(initial_backoff=0.001)
    with pytest.raises(ValueError):
        RetryPolicy(initial_backoff=0)
    with pytest.raises(ValueError):
        RetryPolicy(max_backoff=-1)
    with pytest.raises(ValueError):
        RetryPolicy(max_backoff=float("inf"))
    # Finite, but past what the service config can write: it would overflow building the channel.
    with pytest.raises(ValueError, match="max_backoff is too large to apply"):
        RetryPolicy(max_backoff=1e308)
    # An int too large for a float names the setting rather than leaking OverflowError.
    with pytest.raises(ValueError, match="initial_backoff is out of range"):
        RetryPolicy(initial_backoff=10**1000)
    with pytest.raises(ValueError, match="backoff_multiplier is out of range"):
        RetryPolicy(backoff_multiplier=10**1000)
    # The float nearest the limit in seconds rounds one nanosecond past it; the check is on what is written.
    with pytest.raises(ValueError, match="max_backoff is too large to apply"):
        RetryPolicy(max_backoff=(2**63 - 1) / 1_000_000_000)
    RetryPolicy(max_backoff=9_000_000_000.0)


def test_RetryPolicy_maxBackoffShorterThanInitialBackoff_isRejected() -> None:
    with pytest.raises(ValueError, match="max_backoff must not be shorter than initial_backoff"):
        RetryPolicy(initial_backoff=2, max_backoff=1)


def test_RetryPolicy_multiplierBelowOneOrNotFinite_isRejected() -> None:
    with pytest.raises(ValueError, match="backoff_multiplier must be at least 1, was 0.5"):
        RetryPolicy(backoff_multiplier=0.5)
    with pytest.raises(ValueError):
        RetryPolicy(backoff_multiplier=float("nan"))


def test_RetryPolicy_wrongTypes_areRejected() -> None:
    # A fractional count is no attempt count, and a truthy "no" would otherwise retry every write.
    with pytest.raises(TypeError, match="max_attempts must be an int, was float"):
        RetryPolicy(max_attempts=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="retry_writes must be a bool, was str"):
        RetryPolicy(retry_writes="no")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RetryPolicy(max_attempts=True)
    # True would be written into the service config as JSON true, which gRPC does not take as a number.
    with pytest.raises(TypeError, match="backoff_multiplier must be a number, was bool"):
        RetryPolicy(backoff_multiplier=True)
    with pytest.raises(TypeError, match="initial_backoff must be a number, was str"):
        RetryPolicy(initial_backoff="1")  # type: ignore[arg-type]


def test_ClientOptions_retryPolicyOfTheWrongType_isRejected() -> None:
    with pytest.raises(TypeError, match="retry_policy must be a RetryPolicy, was dict"):
        ClientOptions(retry_policy={"max_attempts": 3})  # type: ignore[arg-type]


def test_retriedMethods_byDefault_coverReadsAndIdempotentCallsButNoWrites() -> None:
    retried = set(retried_methods(RetryPolicy()))

    assert f"{DOCUMENTS}/GetById" in retried
    assert f"{GRAPH}/Traverse" in retried
    # Recall refreshes lastAccessedAt, but repeating it is equivalent to calling it once.
    assert f"{GRAPH}/Recall" in retried
    # The token exchange is what the first call after a wake-up waits on.
    assert f"{TOKENS}/Exchange" in retried

    assert f"{DOCUMENTS}/Upsert" not in retried
    # A statement may write, so it counts as a write.
    assert f"{DOCUMENTS}/ExecuteStatement" not in retried
    # EntityLink adds edges on every call.
    assert f"{GRAPH}/EntityLink" not in retried


def test_retriedMethods_retryingWrites_coverEveryUnaryMethod() -> None:
    retried = retried_methods(RetryPolicy(retry_writes=True))

    assert f"{DOCUMENTS}/Upsert" in retried
    assert f"{GRAPH}/EntityLink" in retried
    documents = data_pb2.DESCRIPTOR.services_by_name["DocumentGrpcService"]
    assert len([name for name in retried if name.startswith(f"{DOCUMENTS}/")]) == len(documents.methods)


def test_serviceConfig_writesTheRetryPolicyAsGrpcReadsIt() -> None:
    config = json.loads(service_config(RetryPolicy(initial_backoff=0.25, max_backoff=4)))

    method_config = config["methodConfig"][0]
    retry_policy  = method_config["retryPolicy"]
    assert retry_policy == {
        "maxAttempts":          5,
        "initialBackoff":       "0.250000000s",
        "maxBackoff":           "4.000000000s",
        "backoffMultiplier":    2.0,
        "retryableStatusCodes": ["UNAVAILABLE"],
    }
    assert {"service": DOCUMENTS, "method": "GetById"} in method_config["name"]


def test_channelArguments_withRetryPolicy_enableRetriesAndCarryTheConfig() -> None:
    arguments = dict(ClientOptions(retry_policy=FAST).channel_arguments())

    assert arguments["grpc.enable_retries"] == 1
    assert json.loads(str(arguments["grpc.service_config"])) == json.loads(service_config(FAST))
    assert "grpc.service_config" not in dict(ClientOptions().channel_arguments())


@dataclass
class FlakyDocuments(data_grpc.DocumentGrpcServiceServicer):
    """A document service that answers ``UNAVAILABLE`` to its first ``failures`` calls."""

    failures: int
    attempts: list[str] = field(default_factory=list)

    def _answer(self, method: str, context: grpc.ServicerContext, value: Any) -> Any:
        self.attempts.append(method)
        if len(self.attempts) <= self.failures:
            context.abort(grpc.StatusCode.UNAVAILABLE, "restarting")
        return value

    def GetById(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        return self._answer("GetById", context, data_pb2.DocumentProto(id=7))

    def Upsert(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        return self._answer("Upsert", context, data_pb2.UpsertDocumentResponse(id=7))


@dataclass
class FlakyTokens(platform_grpc.TokenGrpcServiceServicer):
    """A token service that answers ``UNAVAILABLE`` to its first ``failures`` exchanges."""

    failures: int
    attempts: int = 0

    def Exchange(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self.attempts += 1
        if self.attempts <= self.failures:
            context.abort(grpc.StatusCode.UNAVAILABLE, "waking up")
        return platform_pb2.TokenExchangeResponse(token="a-token", expires_in=3600)


@dataclass
class Flaky:
    port:      int
    documents: FlakyDocuments
    tokens:    FlakyTokens


@pytest.fixture
def flaky() -> Iterator[Any]:
    """Starts a server whose documents and token exchange fail the given number of times first."""
    servers: list[grpc.Server] = []

    def start(document_failures: int = 0, token_failures: int = 0) -> Flaky:
        documents = FlakyDocuments(document_failures)
        tokens    = FlakyTokens(token_failures)
        server    = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        data_grpc.add_DocumentGrpcServiceServicer_to_server(documents, server)
        platform_grpc.add_TokenGrpcServiceServicer_to_server(tokens, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        servers.append(server)
        return Flaky(port, documents, tokens)

    yield start

    for server in servers:
        server.stop(grace=None)


def test_getById_unavailableTwice_succeedsOnTheThirdAttempt(flaky: Any) -> None:
    server = flaky(document_failures=2)
    with CyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=FAST) as client:
        assert client.get_by_id("a-collection", 7).id == 7

    assert server.documents.attempts == ["GetById"] * 3


def test_getById_unavailableOnEveryAttempt_failsAfterMaxAttempts(flaky: Any) -> None:
    server = flaky(document_failures=1_000)
    with (
        CyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=FAST) as client,
        pytest.raises(UnavailableException),
    ):
        client.get_by_id("a-collection", 7)

    assert len(server.documents.attempts) == FAST.max_attempts


def test_upsert_unavailable_isNotRetriedByDefault(flaky: Any) -> None:
    server = flaky(document_failures=1)
    with (
        CyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=FAST) as client,
        pytest.raises(UnavailableException),
    ):
        client.upsert("a-collection", 7)

    # The write may have been applied; it must not be resent.
    assert server.documents.attempts == ["Upsert"]


def test_upsert_unavailableWithRetryWrites_isRetried(flaky: Any) -> None:
    server = flaky(document_failures=1)
    policy = RetryPolicy(initial_backoff=FAST.initial_backoff, max_backoff=FAST.max_backoff, retry_writes=True)
    with CyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=policy) as client:
        assert client.upsert("a-collection", 7) == 7

    assert server.documents.attempts == ["Upsert"] * 2


def test_callTimeout_coversTheRetries(flaky: Any) -> None:
    """With waits of up to ten seconds, a 0.3 s call timeout still ends the call well within seconds."""
    server = flaky(document_failures=1_000)
    slow   = RetryPolicy(initial_backoff=10, max_backoff=10)
    with CyrockDbClient(host="127.0.0.1", port=server.port, call_timeout=0.3, retry_policy=slow) as client:
        started = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 - DEADLINE_EXCEEDED or UNAVAILABLE, depending on the jitter
            client.get_by_id("a-collection", 7)
        elapsed = time.monotonic() - started

    assert elapsed < 5, f"took {elapsed}"


def test_firstCall_tokenExchangeUnavailableTwice_getsItsTokenOnTheThirdAttempt(flaky: Any) -> None:
    """A failed exchange fails the call waiting on it as UNAUTHENTICATED, so the exchange is retried itself."""
    server = flaky(token_failures=2)
    with CyrockDbClient(host="127.0.0.1", port=server.port, api_key="obs_test_key", retry_policy=FAST) as client:
        assert client.get_by_id("a-collection", 7).id == 7

    assert server.tokens.attempts == 3
    assert server.documents.attempts == ["GetById"]


async def test_asyncGetById_unavailableTwice_succeedsOnTheThirdAttempt(flaky: Any) -> None:
    server = flaky(document_failures=2)
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=FAST) as client:
        assert (await client.get_by_id("a-collection", 7)).id == 7

    assert server.documents.attempts == ["GetById"] * 3


def test_firstCall_tokenExchangeNeverAnswers_staysWithinTheCallTimeout(flaky: Any) -> None:
    """The exchange runs before the call and is retried too, and the call timeout still bounds the whole."""
    server = flaky(token_failures=1_000)
    slow   = RetryPolicy(initial_backoff=10, max_backoff=10)
    with CyrockDbClient(
        host="127.0.0.1", port=server.port, api_key="obs_test_key", call_timeout=0.3, retry_policy=slow
    ) as client:
        started = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 - DEADLINE_EXCEEDED or UNAVAILABLE, depending on the jitter
            client.get_by_id("a-collection", 7)
        elapsed = time.monotonic() - started

    assert elapsed < 5, f"took {elapsed}"
    assert server.documents.attempts == []


async def test_asyncGetById_unavailableOnEveryAttempt_failsAfterMaxAttempts(flaky: Any) -> None:
    server = flaky(document_failures=1_000)
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=FAST) as client:
        with pytest.raises(UnavailableException):
            await client.get_by_id("a-collection", 7)

    assert len(server.documents.attempts) == FAST.max_attempts


async def test_asyncUpsert_unavailable_isNotRetriedByDefault(flaky: Any) -> None:
    server = flaky(document_failures=1)
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=FAST) as client:
        with pytest.raises(UnavailableException):
            await client.upsert("a-collection", 7)

    assert server.documents.attempts == ["Upsert"]


async def test_asyncUpsert_unavailableWithRetryWrites_isRetried(flaky: Any) -> None:
    server = flaky(document_failures=1)
    policy = RetryPolicy(initial_backoff=FAST.initial_backoff, max_backoff=FAST.max_backoff, retry_writes=True)
    async with AsyncCyrockDbClient(host="127.0.0.1", port=server.port, retry_policy=policy) as client:
        assert await client.upsert("a-collection", 7) == 7

    assert server.documents.attempts == ["Upsert"] * 2


async def test_asyncCallTimeout_coversTheRetries(flaky: Any) -> None:
    server = flaky(document_failures=1_000)
    slow   = RetryPolicy(initial_backoff=10, max_backoff=10)
    async with AsyncCyrockDbClient(
        host="127.0.0.1", port=server.port, call_timeout=0.3, retry_policy=slow
    ) as client:
        started = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 - DEADLINE_EXCEEDED or UNAVAILABLE, depending on the jitter
            await client.get_by_id("a-collection", 7)
        elapsed = time.monotonic() - started

    assert elapsed < 5, f"took {elapsed}"


async def test_asyncFirstCall_tokenExchangeUnavailableTwice_getsItsTokenOnTheThirdAttempt(flaky: Any) -> None:
    server = flaky(token_failures=2)
    async with AsyncCyrockDbClient(
        host="127.0.0.1", port=server.port, api_key="obs_test_key", retry_policy=FAST
    ) as client:
        assert (await client.get_by_id("a-collection", 7)).id == 7

    assert server.tokens.attempts == 3
    assert server.documents.attempts == ["GetById"]


async def test_asyncFirstCall_tokenExchangeNeverAnswers_staysWithinTheCallTimeout(flaky: Any) -> None:
    server = flaky(token_failures=1_000)
    slow   = RetryPolicy(initial_backoff=10, max_backoff=10)
    async with AsyncCyrockDbClient(
        host="127.0.0.1", port=server.port, api_key="obs_test_key", call_timeout=0.3, retry_policy=slow
    ) as client:
        started = time.monotonic()
        with pytest.raises(Exception):  # noqa: B017 - DEADLINE_EXCEEDED or UNAVAILABLE, depending on the jitter
            await client.get_by_id("a-collection", 7)
        elapsed = time.monotonic() - started

    assert elapsed < 5, f"took {elapsed}"
    assert server.documents.attempts == []
