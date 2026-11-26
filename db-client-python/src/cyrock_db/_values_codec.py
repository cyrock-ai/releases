"""The ``CyrockValue`` wire codec: Python values to and from the shared value type.

The Python counterpart of ``CyrockValues`` (``proto/src/main/java/ai/cyrock/db/proto/value``). This
is the encoding for everything a caller supplies - node, edge and document metadata, CyQL row values
and query parameters - so it is worth stating why the wire type is not
``google.protobuf.Value``: that type inherits JSON's single numeric kind, a double, and this system
offers a ``LONG`` distinct from ``INTEGER``. A LONG past 2^53 does not survive a double, and used to
be rounded silently, with no exception and no warning. That is issue #121, and this codec exists to
make it impossible.

**Two ordering traps, both load-bearing.**

``bool`` is checked before ``int`` because in Python ``bool`` *is* a subclass of ``int``:
``isinstance(True, int)`` is ``True``. Check ints first and ``True`` goes out as ``int_value=1`` and
comes back as ``1``. The Java codec puts ``Boolean`` ahead of its ``Number`` branch for a different
reason - a ``Boolean`` is not a ``Number`` there - but the consequence of getting it wrong is the
same silent type change this whole type exists to prevent.

``str`` and ``bytes`` are checked before the sequence branch because both are sequences in Python. A
string would otherwise be encoded as a list of one-character strings.

**Integers wider than int64.** Python integers are arbitrary precision, so this is routine here
rather than exotic: an ``int`` that does not fit a signed 64-bit field is encoded as its exact
decimal text, matching what the Java codec does with a large ``BigInteger``. Text keeps every digit;
a double would drop most of them. The value reads back as a string, which is lossy in type but not
in magnitude - the alternative is to be wrong about the number itself.

**Temporal values** cross as ISO-8601 text, as the Java codec's ``toString`` arm sends ``LocalDate``,
``LocalTime``, ``LocalDateTime`` and ``Instant``, because that is what the server's
``MetadataFieldType.coerce`` parses back. ``str(datetime)`` is not that text - it separates date and
time with a space, which ``LocalDateTime.parse`` and ``Instant.parse`` both reject - so each type is
encoded explicitly: a naive ``datetime`` as a local date-time (``2024-01-02T10:15:30``, Java's
``LocalDateTime``), a timezone-aware one as the instant it denotes, in UTC with a ``Z``
(``2024-01-02T09:15:30Z``, Java's ``Instant``), a ``date`` as ``LocalDate`` and a naive ``time`` as
``LocalTime``. The text is the one Java's ``toString`` prints, not ``isoformat``'s: the fraction has
3 or 6 digits, as few as hold it (``.123``, not ``.123000``), and a local time or date-time without
seconds stops at the minutes (``10:15``). It matters where the value lands in a ``STRING`` field,
which stores the text as sent. A timezone-aware ``time`` is refused: ``LocalTime`` has no offset and
the server cannot parse one. A ``datetime`` is checked before ``date`` because it is a subclass of
it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any

from ._proto import cyrock_db_value_pb2 as value_pb2

__all__ = ["from_struct", "from_value", "to_struct", "to_value"]

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def to_value(value: Any) -> Any:
    """Encodes one Python value as a ``CyrockValue``."""
    encoded = value_pb2.CyrockValue()

    if value is None:
        # Presence-only: the payload is never read, only the field being set means anything.
        encoded.null_value = True
    elif isinstance(value, bool):
        # Before int. See the module docstring.
        encoded.bool_value = value
    elif isinstance(value, int):
        if INT64_MIN <= value <= INT64_MAX:
            encoded.int_value = value
        else:
            # Wider than int64. Exact text rather than a double that would lose digits.
            encoded.string_value = str(value)
    elif isinstance(value, float):
        encoded.double_value = value
    elif isinstance(value, str):
        # Before the sequence branch: a str is a Sequence.
        encoded.string_value = value
    elif isinstance(value, bytes | bytearray | memoryview):
        # Carried as bytes, not as a base64 string.
        encoded.bytes_value = bytes(value)
    elif isinstance(value, Decimal):
        # Arbitrary precision by definition: neither int64 nor double can be relied on to hold one,
        # so the exact text is the only encoding that does not quietly change the value. "f" is
        # Python's plain-string format - no exponent - matching BigDecimal.toPlainString.
        encoded.string_value = format(value, "f")
    elif isinstance(value, datetime | date | time):
        # Before the fallback below, whose str() would separate date and time with a space.
        encoded.string_value = _temporal_text(value)
    elif isinstance(value, Mapping):
        encoded.struct_value.CopyFrom(to_struct(value))
    elif isinstance(value, Sequence | set | frozenset):
        encoded.list_value.CopyFrom(_to_list(value))
    else:
        # Anything else, as the Java codec's default arm does: something a caller wrote, or a type
        # neither side models. Its text is better than dropping the field.
        encoded.string_value = str(value)

    return encoded


def _temporal_text(value: datetime | date | time) -> str:
    """The ISO-8601 text the Java codec sends for the matching ``java.time`` type. See the module docstring.

    :raises ValueError: if ``value`` is a timezone-aware ``time``, which ``LocalTime`` cannot carry.
    """
    if isinstance(value, datetime):
        # Before date: a datetime is a date.
        if value.utcoffset() is None:
            return value.date().isoformat() + "T" + _java_time_text(value.time(), always_seconds=False)
        instant = value.astimezone(timezone.utc)
        # Instant.toString always prints the seconds, unlike LocalDateTime.
        return instant.date().isoformat() + "T" + _java_time_text(instant.time(), always_seconds=True) + "Z"
    if isinstance(value, time):
        if value.utcoffset() is not None:
            raise ValueError(f"a time with a time zone cannot be sent, the server's TIME has none: {value}")
        return _java_time_text(value, always_seconds=False)
    return value.isoformat()


def _java_time_text(
    value:          time,
    always_seconds: bool,
) -> str:
    """``HH:mm[:ss[.fff|.ffffff]]`` as ``LocalTime.toString`` prints it (seconds always, for an ``Instant``)."""
    text = f"{value.hour:02d}:{value.minute:02d}"
    if not always_seconds and value.second == 0 and value.microsecond == 0:
        return text
    text += f":{value.second:02d}"
    if value.microsecond == 0:
        return text
    if value.microsecond % 1000 == 0:
        return text + f".{value.microsecond // 1000:03d}"
    return text + f".{value.microsecond:06d}"


def _to_list(values: Any) -> Any:
    encoded = value_pb2.CyrockList()
    for element in values:
        encoded.values.append(to_value(element))
    return encoded


def to_struct(mapping: Mapping[Any, Any] | None) -> Any:
    """Encodes a mapping as a ``CyrockStruct``. Keys are stringified, as on the Java side."""
    encoded = value_pb2.CyrockStruct()
    if mapping:
        for key, value in mapping.items():
            encoded.fields[str(key)].CopyFrom(to_value(value))
    return encoded


def from_value(value: Any) -> Any:
    """Decodes a ``CyrockValue``."""
    if value is None:
        return None

    kind = value.WhichOneof("kind")
    if kind is None or kind == "null_value":
        # `None` covers both the explicit null and a kind this build does not know, which is what a
        # newer peer sending a new kind looks like. Not an exception: one unreadable field should not
        # cost the caller the whole message.
        return None
    if kind == "string_value":
        return value.string_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "bytes_value":
        return value.bytes_value
    if kind == "list_value":
        return [from_value(element) for element in value.list_value.values]
    if kind == "struct_value":
        return from_struct(value.struct_value)
    return None


def from_struct(struct: Any) -> dict[str, Any]:
    """Decodes a ``CyrockStruct`` into a plain dict."""
    if struct is None:
        return {}
    return {key: from_value(value) for key, value in struct.fields.items()}
