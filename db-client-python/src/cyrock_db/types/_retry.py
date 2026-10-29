"""How the client retries a call that failed because the server could not be reached.

Mirrors ``ai.cyrock.db.client.java.RetryPolicy``: the same settings, defaults and limits, with the backoffs
in seconds as ``call_timeout`` is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["RetryPolicy"]

MAX_ATTEMPTS_LIMIT         = 5
# The shortest backoff a policy accepts. Below it the waits are too short to give a server any time to come
# back, and grpc-python hangs: it rounds a retry delay down to whole milliseconds, and a delay that comes out
# as zero never fires, so the call waits for its deadline, or forever without one.
MIN_BACKOFF                = 0.01
# The longest backoff the service config can carry, in nanoseconds: it is written in them, and Java stops at
# a signed 64-bit count, about 292 years. Checked on the rounded nanoseconds the config is written from, since
# the float of the limit in seconds rounds one nanosecond past it.
MAX_BACKOFF_NANOS          = 2**63 - 1
DEFAULT_MAX_ATTEMPTS       = MAX_ATTEMPTS_LIMIT
DEFAULT_INITIAL_BACKOFF    = 0.5
DEFAULT_MAX_BACKOFF        = 10.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


def _checked_number(name: str, value: object) -> None:
    # bool is an int to Python, and would be written into the config as JSON true rather than a number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, was {type(value).__name__}")
    # An int too large for a float would raise OverflowError in the range checks below instead of naming the
    # setting; it is out of range for every setting, so say so here.
    try:
        float(value)
    except OverflowError:
        raise ValueError(f"{name} is out of range, was an integer of {len(str(value))} digits") from None


def _checked_backoff(name: str, value: float) -> None:
    _checked_number(name, value)
    if not math.isfinite(value) or value < MIN_BACKOFF:
        raise ValueError(f"{name} must be a finite number of seconds, at least {MIN_BACKOFF}, was {value}")
    # The float comparison first: a value far past the limit would overflow the multiplication to infinity.
    if value > MAX_BACKOFF_NANOS / 1_000_000_000 or round(value * 1_000_000_000) > MAX_BACKOFF_NANOS:
        raise ValueError(f"{name} is too large to apply; the limit is about 292 years, was {value}")


@dataclass(frozen=True)
class RetryPolicy:
    """How the client retries a call that failed because the server could not be reached.

    A backend that is restarting, being moved, or waking up after it was scaled to zero answers
    ``UNAVAILABLE`` until it is back. Without a policy that reaches the caller as
    :class:`~cyrock_db.UnavailableException`, to retry by hand; with one the client retries it itself.
    Pass it as ``retry_policy`` to either client::

        client = CyrockDbClient(api_key="...", call_timeout=30, retry_policy=RetryPolicy())

    **What is retried.** Only calls that failed with ``UNAVAILABLE``, and by default only reads: the
    operations the protocol marks as having no side effects, plus ``recall`` and ``global_search``, which
    touch state but are safe to repeat. A write that fails with ``UNAVAILABLE`` may
    still have been applied, its response lost on the way back, so retrying it can apply it twice - a
    second document, a second edge. ``retry_writes`` opts into that for callers whose writes tolerate it,
    such as upserts by key. CyQL statements count as writes even when they only query, because one
    operation carries both. Change Data Capture subscriptions are never retried here; they reconnect on
    their own. The token exchange of an API-key client is retried too, so an exchange whose answer was lost
    can leave one more ``LOGIN`` event in the audit log; the token from the lost answer goes unused and
    expires.

    **How long it waits.** This is gRPC's own retry. Before retry ``n`` the wait is bounded by
    ``min(initial_backoff * backoff_multiplier ** (n - 1), max_backoff)`` seconds and varied at random, so
    clients that lost the same server do not all come back at the same moment. grpc-python's core library
    varies it by up to a fifth either way around that bound, where the Java client waits a random time
    between zero and the bound; the limits are the same. With a ``call_timeout`` set, that timeout covers
    the call with all its retries: no retry starts after it has run out.

    :param max_attempts: how many times a call is sent in all, the first attempt included: 2 to 5
    :param initial_backoff: seconds; the bound of the wait before the first retry, at least 0.01
    :param max_backoff: seconds; the bound no single wait grows past, at least ``initial_backoff``
    :param backoff_multiplier: how much the upper bound grows after each retry; at least 1
    :param retry_writes: ``True`` to retry calls that change data as well, accepting that one can be
        applied twice
    :raises TypeError: if ``max_attempts`` is not an ``int``, a backoff or the multiplier not a number, or
        ``retry_writes`` not a ``bool``
    :raises ValueError: if a setting is out of range
    """

    max_attempts:       int   = DEFAULT_MAX_ATTEMPTS
    initial_backoff:    float = DEFAULT_INITIAL_BACKOFF
    max_backoff:        float = DEFAULT_MAX_BACKOFF
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    retry_writes:       bool  = False

    def __post_init__(self) -> None:
        # Types first: a fractional count is no attempt count, and a truthy string such as "no" would
        # otherwise switch on retrying every write.
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool):
            raise TypeError(f"max_attempts must be an int, was {type(self.max_attempts).__name__}")
        if not isinstance(self.retry_writes, bool):
            raise TypeError(f"retry_writes must be a bool, was {type(self.retry_writes).__name__}")
        if not 2 <= self.max_attempts <= MAX_ATTEMPTS_LIMIT:
            raise ValueError(f"max_attempts must be between 2 and {MAX_ATTEMPTS_LIMIT}, was {self.max_attempts}")
        _checked_backoff("initial_backoff", self.initial_backoff)
        _checked_backoff("max_backoff", self.max_backoff)
        if self.max_backoff < self.initial_backoff:
            raise ValueError(
                f"max_backoff must not be shorter than initial_backoff, was {self.max_backoff} < {self.initial_backoff}"
            )
        _checked_number("backoff_multiplier", self.backoff_multiplier)
        if not math.isfinite(self.backoff_multiplier) or self.backoff_multiplier < 1.0:
            raise ValueError(f"backoff_multiplier must be at least 1, was {self.backoff_multiplier}")
