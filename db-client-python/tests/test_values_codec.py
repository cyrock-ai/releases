"""The CyrockValue codec, and specifically the ways it can be silently wrong.

Issue #121 is the reason this wire type exists: a LONG past 2^53 was rounded on its way through
``google.protobuf.Value``'s single numeric kind, with no exception and no warning. Most of these
tests are about type and magnitude surviving a round trip, because that is the only property that
matters and the only one whose loss is invisible.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from cyrock_db._values_codec import INT64_MAX, INT64_MIN, from_struct, from_value, to_struct, to_value


def _round_trip(value: Any) -> Any:
    return from_value(to_value(value))


@pytest.mark.parametrize("value", [True, False])
def test_aBool_isEncodedAsABool_notAnInt(value: bool) -> None:
    """The Python-specific trap: bool is a subclass of int, so an int-first check swallows it.

    Getting this wrong is not a crash - ``True`` goes out as 1 and comes back as 1, which is the
    exact class of silent type change this wire type was introduced to stop.
    """
    assert to_value(value).WhichOneof("kind") == "bool_value"
    assert _round_trip(value) is value


def test_aLongPastTheDoubleMantissa_survivesExactly() -> None:
    """The regression issue #121 was filed for. 2^53 + 1 is not representable as a double."""
    value = 2**53 + 1
    assert to_value(value).WhichOneof("kind") == "int_value"
    assert _round_trip(value) == value

    assert _round_trip(INT64_MAX) == INT64_MAX
    assert _round_trip(INT64_MIN) == INT64_MIN


@pytest.mark.parametrize("value", [INT64_MAX + 1, INT64_MIN - 1, 2**200, -(2**200)])
def test_anIntWiderThanInt64_isEncodedAsExactText(value: int) -> None:
    """Python ints are unbounded, so this is routine rather than exotic.

    Lossy in type - it reads back as a string - but not in magnitude, which is the trade the Java
    codec makes for a large BigInteger. A double would keep the type and lose the number.
    """
    assert to_value(value).WhichOneof("kind") == "string_value"
    assert _round_trip(value) == str(value)
    assert int(_round_trip(value)) == value, "every digit survives"


def test_aStr_isNotEncodedAsASequenceOfCharacters() -> None:
    """str is a Sequence, so the sequence branch has to come after it."""
    assert to_value("hello").WhichOneof("kind") == "string_value"
    assert _round_trip("hello") == "hello"


@pytest.mark.parametrize("value", [b"\x00\xff", bytearray(b"ab"), memoryview(b"cd")])
def test_bytesLikeValues_areEncodedAsBytes_notText(value: Any) -> None:
    """bytes is also a Sequence, and base64 text is what the previous encoder did."""
    assert to_value(value).WhichOneof("kind") == "bytes_value"
    assert _round_trip(value) == bytes(value)


def test_aFloat_staysADouble() -> None:
    assert to_value(3.5).WhichOneof("kind") == "double_value"
    assert _round_trip(3.5) == 3.5


def test_none_isEncodedAsThePresenceOnlyNull() -> None:
    assert to_value(None).WhichOneof("kind") == "null_value"
    assert _round_trip(None) is None


def test_aDecimal_isEncodedAsPlainText_withNoExponent() -> None:
    """Arbitrary precision by definition; neither int64 nor double can be relied on to hold one."""
    assert _round_trip(Decimal("1.2345678901234567890123")) == "1.2345678901234567890123"
    assert "E" not in _round_trip(Decimal("1E+20")), "plain string, not scientific notation"


@pytest.mark.parametrize(
    ("value", "expected"),
    [([1, "x", True], [1, "x", True]), ((1, 2), [1, 2]), ({"a": 1}, {"a": 1})],
)
def test_containers_roundTripElementByElement(value: Any, expected: Any) -> None:
    assert _round_trip(value) == expected


def test_nesting_isPreservedToAnyDepth() -> None:
    value = {"outer": {"inner": [1, {"deep": True}, None]}}
    assert _round_trip(value) == value


def test_aBoolInsideAContainer_isStillABool() -> None:
    """The ordering trap has to hold in the recursive case too, not only at the top level."""
    assert _round_trip({"flag": True})["flag"] is True
    assert _round_trip([False])[0] is False


def test_toStruct_stringifiesKeys_asTheJavaCodecDoes() -> None:
    assert from_struct(to_struct({1: "a", None: "b"})) == {"1": "a", "None": "b"}


def test_toStruct_none_isAnEmptyStruct() -> None:
    assert from_struct(to_struct(None)) == {}


def test_fromValue_anUnsetKind_readsAsNone() -> None:
    """What a newer peer sending a kind this build does not know looks like.

    None rather than an exception: one unreadable field should not cost the caller the whole message.
    """
    from cyrock_db._proto import cyrock_db_value_pb2 as value_pb2

    assert from_value(value_pb2.CyrockValue()) is None
    assert from_value(None) is None


def test_anUnmodelledType_fallsBackToItsText() -> None:
    class Custom:
        def __str__(self) -> str:
            return "custom-repr"

    assert _round_trip(Custom()) == "custom-repr"


_PLUS_TWO = timezone(timedelta(hours=2))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2024, 1, 2, 10, 15, 30),                         "2024-01-02T10:15:30"),
        (datetime(2024, 1, 2, 10, 15, 30, 123000),                 "2024-01-02T10:15:30.123"),
        (datetime(2024, 1, 2, 10, 15, 30, 123400),                 "2024-01-02T10:15:30.123400"),
        (datetime(2024, 1, 2, 10, 15),                             "2024-01-02T10:15"),
        (datetime(2024, 1, 2, 10, 15, 0, 500000),                  "2024-01-02T10:15:00.500"),
        (datetime(2024, 1, 2, 10, 15, 30, tzinfo=timezone.utc),    "2024-01-02T10:15:30Z"),
        (datetime(2024, 1, 2, 10, 15, 30, tzinfo=_PLUS_TWO),       "2024-01-02T08:15:30Z"),
        (datetime(2024, 1, 2, 10, 15, tzinfo=timezone.utc),        "2024-01-02T10:15:00Z"),
        (datetime(2024, 1, 2, 10, 15, 30, 120000, tzinfo=_PLUS_TWO), "2024-01-02T08:15:30.120Z"),
        (date(2024, 1, 2),                                         "2024-01-02"),
        (time(10, 15, 30),                                         "10:15:30"),
        (time(10, 15),                                             "10:15"),
        (time(10, 15, 30, 7),                                      "10:15:30.000007"),
    ],
)
def test_aTemporal_isEncodedAsTheIsoTextTheJavaCodecSends(value: Any, expected: str) -> None:
    """``str(datetime)`` separates date and time with a space, which the server cannot parse.

    The expected texts are what ``LocalDateTime``, ``Instant``, ``LocalDate`` and ``LocalTime`` print in
    Java, and what ``MetadataFieldType.coerce`` parses back: a naive datetime is a local date-time, an
    aware one the instant it denotes, normalised to UTC.
    """
    encoded = to_value(value)

    assert encoded.WhichOneof("kind") == "string_value"
    assert encoded.string_value == expected


def test_aTemporalInsideAContainer_isEncodedAsIsoTextToo() -> None:
    assert _round_trip({"at": [datetime(2024, 1, 2, 10, 15)]}) == {"at": ["2024-01-02T10:15"]}


def test_aTimeWithATimeZone_isRefused() -> None:
    """``LocalTime`` has no offset, so the server cannot parse ``10:15:00+02:00``."""
    with pytest.raises(ValueError, match="time zone"):
        to_value(time(10, 15, tzinfo=_PLUS_TWO))
