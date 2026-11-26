"""Python client for the CYROCK.AI DB unified AI data engine.

The blocking client and its change-stream handles are exported here: :class:`ChangeListener` is what
``watch_graph`` and ``watch_collection`` take, and :class:`ChangeStreamSubscription` what they return. The
asyncio client is :class:`cyrock_db.aio.AsyncCyrockDbClient`, and the domain types are in :mod:`cyrock_db.types`.

Platform administration (projects and API keys) is the separate :class:`CyrockDbAdminClient`, with its asyncio
twin :class:`cyrock_db.aio.AsyncCyrockDbAdminClient`.
"""

from __future__ import annotations

from ._admin import CyrockDbAdminClient
from ._watch import ChangeListener, ChangeStreamSubscription
from .client import CyrockDbClient
from .exceptions import (
    CyrockDbClientException,
    DeadlineExceededException,
    DurabilityNotConfirmedException,
    ForbiddenException,
    NotFoundException,
    RetryableException,
    StorageFullException,
    StoreBusyException,
    UnauthorizedException,
    UnavailableException,
    WrongCredentialException,
)

__all__ = [
    "ChangeListener",
    "ChangeStreamSubscription",
    "CyrockDbAdminClient",
    "CyrockDbClient",
    "CyrockDbClientException",
    "DeadlineExceededException",
    "DurabilityNotConfirmedException",
    "ForbiddenException",
    "NotFoundException",
    "RetryableException",
    "StorageFullException",
    "StoreBusyException",
    "UnauthorizedException",
    "UnavailableException",
    "WrongCredentialException",
]
