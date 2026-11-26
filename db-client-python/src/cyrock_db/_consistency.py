"""The consistent-read request header.

The counterpart of ``client-java``'s ``ReadConsistencyHeader``, which the data server reads. Several data
servers may serve one storage, and a data server learns of another one's commit within one notification
latency. A call carrying this header first asks the storage whether another data server committed to the
collection or graph it reads, and so sees every write that returned before it was sent, whichever data server
took it. See :class:`~cyrock_db._channel.ClientOptions` ``consistent_reads``.
"""

from __future__ import annotations

__all__ = ["CONSISTENT", "CONSISTENT_READ_HEADER", "consistent_read_header"]

CONSISTENT_READ_HEADER = "x-cyrock-consistent-read"
CONSISTENT             = "true"


def consistent_read_header() -> tuple[str, str]:
    """The header that asks for a consistent read."""
    return (CONSISTENT_READ_HEADER, CONSISTENT)
