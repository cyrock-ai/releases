"""Request building and response mapping of the admin clients, without a server.

The counterpart of client-java's ``AdminProtoMappingTest``, case for case, and of ``AdminTypesTest`` for the
validation the types do on construction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cyrock_db import _proto_converters as converters
from cyrock_db import _proto_requests as requests
from cyrock_db._proto import cyrock_db_platform_pb2 as platform_pb2
from cyrock_db.exceptions import CyrockDbClientException
from cyrock_db.types import ApiKeyInfo, ApiKeyStatus, CreatedApiKey, NewApiKeyRequest, ProjectInfo

WHEN = datetime(2026, 11, 20, 10, 15, 30, 123456, tzinfo=timezone.utc)


# ─── Requests ──────────────────────────────────────────────────────────────


def test_createProject_arguments_areSetOnTheRequest() -> None:
    request = requests.create_project("org-1", "support")

    assert (request.organization_id, request.name) == ("org-1", "support")


def test_createApiKey_fullRequest_setsEveryField() -> None:
    request = requests.create_api_key(NewApiKeyRequest(
        organization_id = "org-1",
        name            = "app",
        project_id      = "p-1",
        user_id         = "u-1",
        permissions     = ("DOCUMENTS_READ",),
        expires_at      = WHEN,
    ))

    assert (request.organization_id, request.name, request.project_id, request.user_id) == (
        "org-1", "app", "p-1", "u-1"
    )
    assert list(request.permissions) == ["DOCUMENTS_READ"]
    assert request.expires_at.ToDatetime(tzinfo=timezone.utc) == WHEN


def test_createApiKey_expiryInAnotherZone_isSentAsTheSameInstant() -> None:
    elsewhere = WHEN.astimezone(timezone(timedelta(hours=5)))

    request = requests.create_api_key(NewApiKeyRequest(organization_id="o", name="n", expires_at=elsewhere))

    assert request.expires_at.ToDatetime(tzinfo=timezone.utc) == WHEN


def test_createApiKey_minimalRequest_leavesTheOptionalFieldsUnset() -> None:
    request = requests.create_api_key(NewApiKeyRequest(organization_id="org-1", name="app"))

    assert (request.project_id, request.user_id, list(request.permissions)) == ("", "", [])
    assert not request.HasField("expires_at"), "an unset expiry must stay absent, or the key would expire in 1970"


# ─── Responses ─────────────────────────────────────────────────────────────


def test_toProjectInfo_fullProto_mapsEveryField() -> None:
    proto = platform_pb2.ProjectProto(id="p-1", organization_id="org-1", name="support", enabled=True)
    proto.created_at.FromDatetime(WHEN)

    assert converters.to_project_info(proto) == ProjectInfo(
        id="p-1", organization_id="org-1", name="support", created_at=WHEN, enabled=True
    )


def test_toProjectInfo_noCreatedAt_readsItAsNone() -> None:
    assert converters.to_project_info(platform_pb2.ProjectProto(id="p")).created_at is None


def test_toApiKeyInfo_revokedKey_mapsEveryField() -> None:
    proto = platform_pb2.ApiKeyProto(
        id="k-1", name="app", user_id="u-1", organization_id="org-1", project_id="p-1",
        permissions=["DOCUMENTS_READ", "DOCUMENTS_SEARCH"], status="REVOKED",
    )
    proto.created_at.FromDatetime(WHEN)
    proto.expires_at.FromDatetime(WHEN + timedelta(seconds=1))
    proto.last_used_at.FromDatetime(WHEN + timedelta(seconds=2))
    proto.revoked_at.FromDatetime(WHEN + timedelta(seconds=3))

    assert converters.to_api_key_info(proto) == ApiKeyInfo(
        id              = "k-1",
        name            = "app",
        user_id         = "u-1",
        organization_id = "org-1",
        project_id      = "p-1",
        permissions     = frozenset({"DOCUMENTS_READ", "DOCUMENTS_SEARCH"}),
        status          = ApiKeyStatus.REVOKED,
        created_at      = WHEN,
        expires_at      = WHEN + timedelta(seconds=1),
        last_used_at    = WHEN + timedelta(seconds=2),
        revoked_at      = WHEN + timedelta(seconds=3),
    )


def test_toApiKeyInfo_organizationWideKeyOfNoUser_readsTheEmptyIdsAsNone() -> None:
    key = converters.to_api_key_info(platform_pb2.ApiKeyProto(id="k", name="n", organization_id="o", status="ACTIVE"))

    assert (key.user_id, key.project_id, key.expires_at, key.last_used_at, key.revoked_at) == (None,) * 5


def test_toApiKeyInfo_statusANewerPlatformAdded_readsAsUnknown() -> None:
    proto = platform_pb2.ApiKeyProto(id="k", name="n", organization_id="o", status="SUSPENDED")

    assert converters.to_api_key_info(proto).status is ApiKeyStatus.UNKNOWN


def test_toApiKeyInfo_noStatusOnTheWire_readsAsUnknown() -> None:
    proto = platform_pb2.ApiKeyProto(id="k", name="n", organization_id="o")

    assert converters.to_api_key_info(proto).status is ApiKeyStatus.UNKNOWN


def test_toApiKeyInfo_expiryPastYear9999_isClampedSoTheListingStaysReadable() -> None:
    # A key the platform stored before it refused such an expiry: datetime cannot hold year 10000.
    proto = platform_pb2.ApiKeyProto(id="k", name="n", organization_id="o", status="ACTIVE")
    proto.expires_at.seconds = 253402300800
    proto.created_at.seconds = -62135596801

    key = converters.to_api_key_info(proto)

    assert key.expires_at == datetime.max.replace(tzinfo=timezone.utc)
    assert key.created_at == datetime.min.replace(tzinfo=timezone.utc)


def test_toCreatedApiKey_withSecret_returnsIt() -> None:
    proto = platform_pb2.ApiKeyCreatedResponse(
        api_key=platform_pb2.ApiKeyProto(id="k", name="n", organization_id="o", status="ACTIVE"),
        plaintext_secret="s3cret",
    )

    assert converters.to_created_api_key(proto).secret == "s3cret"


def test_toCreatedApiKey_withoutSecret_raisesNamingTheKey() -> None:
    proto = platform_pb2.ApiKeyCreatedResponse(
        api_key=platform_pb2.ApiKeyProto(id="k-9", name="n", organization_id="o", status="ACTIVE"),
    )

    with pytest.raises(CyrockDbClientException, match="k-9") as raised:
        converters.to_created_api_key(proto)
    assert raised.value.status_code == 500


# ─── Types ─────────────────────────────────────────────────────────────────


def test_newApiKeyRequest_blankOptionalIds_areReadAsAbsent() -> None:
    request = NewApiKeyRequest(organization_id="o", name="n", project_id=" ", user_id="")

    assert (request.project_id, request.user_id) == (None, None)


@pytest.mark.parametrize(("organization_id", "name"), [(" ", "app"), ("org-1", "")])
def test_newApiKeyRequest_blankOrganizationOrName_isRejected(organization_id: str, name: str) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        NewApiKeyRequest(organization_id=organization_id, name=name)


def test_newApiKeyRequest_blankPermission_isRejected() -> None:
    with pytest.raises(ValueError, match="permission must not be blank"):
        NewApiKeyRequest(organization_id="o", name="n", permissions=(" ",))


def test_newApiKeyRequest_permissionsAsAString_isRejected() -> None:
    with pytest.raises(ValueError, match="collection of names"):
        NewApiKeyRequest(organization_id="o", name="n", permissions="DOCUMENTS_READ")  # type: ignore[arg-type]


def test_newApiKeyRequest_duplicatePermissions_keepTheFirstOrder() -> None:
    request = NewApiKeyRequest(organization_id="o", name="n", permissions=("B", "A", "B"))

    assert request.permissions == ("B", "A")


def test_newApiKeyRequest_naiveExpiry_isRejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        NewApiKeyRequest(organization_id="o", name="n", expires_at=datetime(2027, 1, 1))  # noqa: DTZ001


@pytest.mark.parametrize("expires_at", [datetime(1970, 1, 1, tzinfo=timezone.utc),
                                        datetime(1969, 12, 31, tzinfo=timezone.utc)])
def test_newApiKeyRequest_expiryAtOrBeforeTheEpoch_isRejected(expires_at: datetime) -> None:
    # The platform reads a zero timestamp as "no expiry", so the epoch would turn into a key that never expires.
    with pytest.raises(ValueError, match="after the epoch"):
        NewApiKeyRequest(organization_id="o", name="n", expires_at=expires_at)


def test_newApiKeyRequest_expiryPastYear9999InUtc_isRejected() -> None:
    # 23:59 on the last day of 9999 at UTC-05:00 is 04:59 on 10000-01-01 in UTC: past the Timestamp range.
    late = datetime(9999, 12, 31, 23, 59, tzinfo=timezone(timedelta(hours=-5)))

    with pytest.raises(ValueError, match="at most"):
        NewApiKeyRequest(organization_id="o", name="n", expires_at=late)


def test_newApiKeyRequest_lastRepresentableExpiry_isAccepted() -> None:
    last = datetime.max.replace(tzinfo=timezone.utc)

    assert NewApiKeyRequest(organization_id="o", name="n", expires_at=last).expires_at == last


def test_createApiKey_lastRepresentableExpiry_isSentWithinTheTimestampRange() -> None:
    request = requests.create_api_key(
        NewApiKeyRequest(organization_id="o", name="n", expires_at=datetime.max.replace(tzinfo=timezone.utc))
    )

    assert request.expires_at.seconds == 253402300799


def test_apiKeyInfo_permissions_areAFrozenSet() -> None:
    key = ApiKeyInfo(
        id="k", name="n", user_id=None, organization_id="o", project_id=None,
        permissions=frozenset({"DOCUMENTS_READ"}), status=ApiKeyStatus.ACTIVE,
    )

    assert isinstance(key.permissions, frozenset)


def test_createdApiKey_repr_leavesTheSecretOut() -> None:
    key = ApiKeyInfo(
        id="k", name="n", user_id=None, organization_id="o", project_id=None,
        permissions=frozenset(), status=ApiKeyStatus.ACTIVE,
    )

    created = CreatedApiKey(api_key=key, secret="s3cret")

    assert "s3cret" not in repr(created)
    assert "s3cret" not in str(created)


def test_createdApiKey_emptySecret_isRejected() -> None:
    key = ApiKeyInfo(
        id="k", name="n", user_id=None, organization_id="o", project_id=None,
        permissions=frozenset(), status=ApiKeyStatus.ACTIVE,
    )

    with pytest.raises(ValueError, match="secret must not be empty"):
        CreatedApiKey(api_key=key, secret="")
