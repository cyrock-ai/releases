"""Platform administration types: projects and API keys.

Mirrors ``ProjectInfo``, ``ApiKeyInfo``, ``ApiKeyStatus``, ``NewApiKeyRequest`` and ``CreatedApiKey`` from the
``data`` module, which the administration clients (``CyrockDbAdminClient``) take and return.

Timestamps are timezone-aware :class:`~datetime.datetime` values in UTC, where Java has ``Instant``; an absent
one is ``None``. Permission sets are ``frozenset`` where Java has an unmodifiable ``Set``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

__all__ = [
    "MAX_EXPIRES_AT",
    "ApiKeyInfo",
    "ApiKeyStatus",
    "CreatedApiKey",
    "NewApiKeyRequest",
    "ProjectInfo",
]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

MAX_EXPIRES_AT = datetime.max.replace(tzinfo=timezone.utc)
"""The latest expiry a key can have, as Java's ``NewApiKeyRequest.MAX_EXPIRES_AT``: the end of year 9999 in UTC, the
end of the protobuf ``Timestamp`` range. For a key that never expires, leave ``expires_at`` as ``None``."""


class ApiKeyStatus(Enum):
    """The lifecycle state of an API key, as the platform reports it by name."""

    ACTIVE  = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    """Mirrors the platform status of that name, which the platform does not set today: a key past its expiry is
    reported ``ACTIVE`` with ``expires_at`` in the past, and authenticates nothing."""
    UNKNOWN = "UNKNOWN"
    """A status this SDK version does not know, which a newer platform reports: the SDK's own member, never sent by
    the platform. It keeps a listing from a newer platform readable; upgrade the SDK to read the status itself."""


def _require_text(component: str, value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{component} must not be blank")


def _require_aware(component: str, value: datetime | None) -> None:
    # A naive datetime has no instant: read as local time on one machine and as UTC on the server, it would name two
    # different moments. Java's Instant cannot be ambiguous, so neither may this.
    if value is not None and value.utcoffset() is None:
        raise ValueError(f"{component} must be timezone-aware, was {value!r}")


@dataclass(frozen=True)
class ProjectInfo:
    """A project as the platform reports it to an administrator.

    :param id: the project's id
    :param organization_id: the organization that owns the project
    :param name: the project's display name
    :param created_at: when the project was created, or ``None`` if the platform did not report it
    :param enabled: whether the project is enabled
    """

    id:              str
    organization_id: str
    name:            str
    created_at:      datetime | None = None
    enabled:         bool            = True

    def __post_init__(self) -> None:
        _require_aware("created_at", self.created_at)


@dataclass(frozen=True)
class ApiKeyInfo:
    """An API key as the platform lists it: everything about it except its secret.

    The platform stores only a hash of a key's secret, so no listing can show it; it is handed out once, as
    :attr:`CreatedApiKey.secret`.

    :param id: the key's id. Not the secret; it authenticates nothing
    :param name: the key's display name
    :param user_id: the user the key acts for, or ``None`` for a key attributed to no user
    :param organization_id: the organization the key belongs to
    :param project_id: the project the key is pinned to, or ``None`` for an organization-wide key
    :param permissions: the permission names the key was granted, such as ``DOCUMENTS_READ``
    :param status: the key's lifecycle state
    :param created_at: when the key was created, or ``None`` if the platform did not report it
    :param expires_at: when the key expires, or ``None`` if it does not
    :param last_used_at: when the key last authenticated, or ``None`` if it never has
    :param revoked_at: when the key was revoked, or ``None`` if it has not been
    """

    id:              str
    name:            str
    user_id:         str | None
    organization_id: str
    project_id:      str | None
    permissions:     frozenset[str]
    status:          ApiKeyStatus
    created_at:      datetime | None = None
    expires_at:      datetime | None = None
    last_used_at:    datetime | None = None
    revoked_at:      datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "permissions", frozenset(self.permissions))
        if not isinstance(self.status, ApiKeyStatus):
            raise ValueError(f"status must be an ApiKeyStatus, was {self.status!r}")
        for component in ("created_at", "expires_at", "last_used_at", "revoked_at"):
            _require_aware(component, getattr(self, component))


@dataclass(frozen=True)
class NewApiKeyRequest:
    """What an API key to create should be: its name, its tenant, the user it acts for and what it may do.

    The platform bounds every key twice, so a request asking for more is refused rather than narrowed: a key cannot
    be granted a permission the creating key does not hold, and a key attributed to a user cannot be granted one
    that user does not hold where the key acts. A key pinned to a project acts in that project only; one without a
    project acts across the organization.

    A blank ``project_id`` or ``user_id`` is read as absent. The permissions keep the order given.

    :param organization_id: the organization the key belongs to
    :param name: the key's display name
    :param project_id: the project the key is pinned to, or ``None`` for an organization-wide key. A creating key
        that is itself pinned to a project may only create keys pinned to that same project
    :param user_id: the user the key acts for, or ``None`` for none. A key attributed to a user is bounded by that
        user's membership on every call
    :param permissions: the permission names to grant, such as ``DOCUMENTS_READ``
    :param expires_at: when the key stops authenticating, timezone-aware, after the epoch and at most
        :data:`MAX_EXPIRES_AT`, or ``None`` for a key that does not expire
    :raises ValueError: if the organization, the name or a permission name is blank, or ``expires_at`` is naive, not
        after the epoch or after :data:`MAX_EXPIRES_AT`
    """

    organization_id: str
    name:            str
    project_id:      str | None      = None
    user_id:         str | None      = None
    permissions:     tuple[str, ...] = ()
    expires_at:      datetime | None = None

    def __post_init__(self) -> None:
        _require_text("organization_id", self.organization_id)
        _require_text("name", self.name)
        object.__setattr__(self, "project_id", _blank_to_none(self.project_id))
        object.__setattr__(self, "user_id", _blank_to_none(self.user_id))
        permissions = tuple(dict.fromkeys(_permissions(self.permissions)))
        for permission in permissions:
            _require_text("permission", permission)
        object.__setattr__(self, "permissions", permissions)
        _require_aware("expires_at", self.expires_at)
        # The platform reads a zero timestamp as no expiry, so the epoch would silently become a key that never
        # expires; an instant before it names no sensible expiry either.
        if self.expires_at is not None and self.expires_at <= _EPOCH:
            raise ValueError(f"expires_at must be after the epoch, was {self.expires_at.isoformat()}")
        # A datetime cannot pass year 9999 in its own zone, but in a zone behind UTC it can name an instant past
        # the end of 9999 in UTC, beyond the protobuf Timestamp range. Compared as instants, which cannot overflow.
        if self.expires_at is not None and self.expires_at > MAX_EXPIRES_AT:
            raise ValueError(
                f"expires_at must be at most {MAX_EXPIRES_AT.isoformat()}, was {self.expires_at.isoformat()}"
            )


def _blank_to_none(value: str | None) -> str | None:
    return value if value is not None and value.strip() else None


def _permissions(value: Iterable[str]) -> Iterable[str]:
    # A bare string is iterable too, and would grant one permission per character.
    if isinstance(value, str):
        raise ValueError(f"permissions must be a collection of names, not the string {value!r}")
    return value


@dataclass(frozen=True)
class CreatedApiKey:
    """A newly created API key together with its secret, which the platform shows this once.

    The platform keeps only a hash of the secret, so nothing can show it again: store it now. A lost secret is
    replaced by creating a new key and revoking the old one.

    Where Java hands out a zeroable ``char[]``, this is a ``str``: Python strings are immutable and cannot be zeroed,
    so there is nothing to gain from another type. The secret is left out of ``repr``, so printing or logging this
    object never shows it.

    :param api_key: the created key, as a listing would show it
    :param secret: the key's plaintext secret: what a client sends as its API key
    :raises ValueError: if the secret is empty
    """

    api_key: ApiKeyInfo
    secret:  str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.secret:
            raise ValueError("secret must not be empty")
