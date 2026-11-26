"""Both admin facades against real platform administration servicers on a loopback port.

The Python counterpart of ``CyrockDbAdminClientTest``. The server refuses a call without ``x-api-key``, per
``credential-by-service.tsv``, so these tests assert the credential as well as the request each operation sends,
the response it maps and the failure it raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent import futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import grpc
import pytest

from conftest import CREDENTIAL_CONTRACT, CredentialEnforcingInterceptor, load_credential_contract
from cyrock_db import CyrockDbAdminClient
from cyrock_db._proto import cyrock_db_platform_pb2 as platform_pb2
from cyrock_db._proto import cyrock_db_platform_pb2_grpc as platform_grpc
from cyrock_db.aio import AsyncCyrockDbAdminClient
from cyrock_db.exceptions import ForbiddenException, NotFoundException
from cyrock_db.types import ApiKeyStatus, NewApiKeyRequest, ProjectInfo

API_KEY = "obs_admin_key"


def _project(project_id: str, name: str = "support") -> Any:
    return platform_pb2.ProjectProto(id=project_id, organization_id="org-1", name=name, enabled=True)


def _api_key(status: str) -> Any:
    return platform_pb2.ApiKeyProto(
        id="k-1", name="app", organization_id="org-1", permissions=["DOCUMENTS_READ"], status=status
    )


@dataclass
class FakeProjects(platform_grpc.ProjectGrpcServiceServicer):
    """Answers as the platform's project service would, recording what it was sent and with which key."""

    created:  list[Any] = field(default_factory=list)
    api_keys: list[str] = field(default_factory=list)

    def _record_key(self, context: grpc.ServicerContext) -> None:
        self.api_keys.append(str(dict(context.invocation_metadata()).get("x-api-key")))

    def Create(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._record_key(context)
        self.created.append(request)
        return _project("p-1", request.name)

    def List(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._record_key(context)
        return platform_pb2.ProjectList(projects=[_project("p-1"), _project("p-2")])

    def Delete(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._record_key(context)
        if request.id != "p-1":
            context.abort(grpc.StatusCode.NOT_FOUND, "Project not found")
        return _project("p-1")


@dataclass
class FakeApiKeys(platform_grpc.ApiKeyGrpcServiceServicer):
    """Answers as the platform's API key service would, recording what it was sent."""

    created:    list[Any] = field(default_factory=list)
    revoked:    list[str] = field(default_factory=list)
    far_future: bool      = False

    def Create(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self.created.append(request)
        return platform_pb2.ApiKeyCreatedResponse(api_key=_api_key("ACTIVE"), plaintext_secret="the-secret")

    def List(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        if self.far_future:
            stored = _api_key("ACTIVE")
            stored.expires_at.seconds = 253402300800  # 10000-01-01, past what a Python datetime can hold
            return platform_pb2.ApiKeyList(api_keys=[stored, _api_key("ACTIVE")])
        return platform_pb2.ApiKeyList(api_keys=[_api_key("ACTIVE")])

    def Delete(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        if request.id == "k-suspended":
            self.revoked.append(request.id)
            return _api_key("SUSPENDED")
        if request.id != "k-1":
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Requires API_KEYS_REVOKE")
        self.revoked.append(request.id)
        return _api_key("REVOKED")


@dataclass
class AdminServer:
    port:     int
    projects: FakeProjects
    api_keys: FakeApiKeys


@pytest.fixture
def admin_server() -> Iterator[AdminServer]:
    projects = FakeProjects()
    api_keys = FakeApiKeys()
    server   = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[CredentialEnforcingInterceptor(load_credential_contract())],
    )
    platform_grpc.add_ProjectGrpcServiceServicer_to_server(projects, server)
    platform_grpc.add_ApiKeyGrpcServiceServicer_to_server(api_keys, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield AdminServer(port=port, projects=projects, api_keys=api_keys)
    server.stop(grace=None)


def _admin(server: AdminServer) -> CyrockDbAdminClient:
    return CyrockDbAdminClient(host="127.0.0.1", port=server.port, api_key=API_KEY)


def _async_admin(server: AdminServer) -> AsyncCyrockDbAdminClient:
    return AsyncCyrockDbAdminClient(host="127.0.0.1", port=server.port, api_key=API_KEY)


# ─── Blocking ──────────────────────────────────────────────────────────────


def test_createProject_validArguments_sendsThemWithTheKeyAndMapsTheProject(admin_server: AdminServer) -> None:
    with _admin(admin_server) as admin:
        project = admin.create_project("org-1", "support")

    assert admin_server.projects.created[0].organization_id == "org-1"
    assert admin_server.projects.created[0].name == "support"
    assert admin_server.projects.api_keys == [API_KEY]
    assert project == ProjectInfo(id="p-1", organization_id="org-1", name="support", created_at=None, enabled=True)


def test_listProjects_twoProjects_returnsBothInOrder(admin_server: AdminServer) -> None:
    with _admin(admin_server) as admin:
        assert [project.id for project in admin.list_projects()] == ["p-1", "p-2"]


def test_deleteProject_unknownProject_raisesNotFound(admin_server: AdminServer) -> None:
    with _admin(admin_server) as admin, pytest.raises(NotFoundException) as raised:
        admin.delete_project("missing")
    assert raised.value.status_code == 404


def test_createApiKey_fullRequest_sendsItAndReturnsTheSecret(admin_server: AdminServer) -> None:
    with _admin(admin_server) as admin:
        created = admin.create_api_key(
            NewApiKeyRequest(organization_id="org-1", name="app", project_id="p-1", permissions=("DOCUMENTS_READ",))
        )

    sent = admin_server.api_keys.created[0]
    assert (sent.organization_id, sent.project_id, list(sent.permissions)) == ("org-1", "p-1", ["DOCUMENTS_READ"])
    assert created.secret == "the-secret"
    assert created.api_key.status is ApiKeyStatus.ACTIVE
    assert "the-secret" not in repr(created)


def test_listApiKeys_oneKey_mapsItWithoutASecret(admin_server: AdminServer) -> None:
    with _admin(admin_server) as admin:
        keys = admin.list_api_keys()

    assert [key.id for key in keys] == ["k-1"]
    assert keys[0].user_id is None


def test_revokeApiKey_activeKey_returnsItRevoked(admin_server: AdminServer) -> None:
    with _admin(admin_server) as admin:
        revoked = admin.revoke_api_key("k-1")

    assert admin_server.api_keys.revoked == ["k-1"]
    assert revoked.status is ApiKeyStatus.REVOKED


def test_listApiKeys_aStoredExpiryPastYear9999_staysReadable(admin_server: AdminServer) -> None:
    admin_server.api_keys.far_future = True

    with _admin(admin_server) as admin:
        keys = admin.list_api_keys()

    assert len(keys) == 2
    assert keys[0].expires_at == datetime.max.replace(tzinfo=timezone.utc)


def test_revokeApiKey_keyInAStatusANewerPlatformAdded_returnsItAsUnknownWithoutRaising(
    admin_server: AdminServer,
) -> None:
    with _admin(admin_server) as admin:
        revoked = admin.revoke_api_key("k-suspended")

    assert admin_server.api_keys.revoked == ["k-suspended"], "the revoke happened"
    assert revoked.status is ApiKeyStatus.UNKNOWN


def test_revokeApiKey_permissionDenied_raisesForbidden(admin_server: AdminServer) -> None:
    with _admin(admin_server) as admin, pytest.raises(ForbiddenException):
        admin.revoke_api_key("not-yours")


@pytest.mark.skipif(
    not CREDENTIAL_CONTRACT.is_file(),
    reason="the proto module is not beside this checkout, so the test server enforces no credential",
)
def test_anyCall_withoutAnApiKeyOnTheWire_isRefusedUnauthenticated(admin_server: AdminServer) -> None:
    """The enforcing server notices a missing key, so the tests above assert the credential, not only the request."""
    with grpc.insecure_channel(f"127.0.0.1:{admin_server.port}") as channel, pytest.raises(grpc.RpcError) as raised:
        platform_grpc.ProjectGrpcServiceStub(channel).List(platform_pb2.Empty())  # type: ignore[no-untyped-call]
    assert raised.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_constructor_withoutAnApiKey_raisesValueError() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        CyrockDbAdminClient()
    with pytest.raises(ValueError, match="api_key is required"):
        CyrockDbAdminClient(api_key=" ")


def test_constructor_withConsistentReads_raisesTypeError() -> None:
    with pytest.raises(TypeError, match="consistent_reads"):
        CyrockDbAdminClient(api_key=API_KEY, consistent_reads=True)


def test_close_calledTwice_isIdempotent() -> None:
    admin = CyrockDbAdminClient(api_key=API_KEY, port=1)

    admin.close()
    admin.close()


# ─── asyncio ───────────────────────────────────────────────────────────────


async def test_asyncCreateProject_validArguments_sendsThemWithTheKey(admin_server: AdminServer) -> None:
    async with _async_admin(admin_server) as admin:
        project = await admin.create_project("org-1", "support")

    assert project.id == "p-1"
    assert admin_server.projects.api_keys == [API_KEY]


async def test_asyncRoundOfOperations_mapsEveryAnswer(admin_server: AdminServer) -> None:
    async with _async_admin(admin_server) as admin:
        projects = await admin.list_projects()
        created  = await admin.create_api_key(NewApiKeyRequest(organization_id="org-1", name="app"))
        keys     = await admin.list_api_keys()
        revoked  = await admin.revoke_api_key("k-1")
        deleted  = await admin.delete_project("p-1")

    assert [project.id for project in projects] == ["p-1", "p-2"]
    assert created.secret == "the-secret"
    assert [key.id for key in keys] == ["k-1"]
    assert revoked.status is ApiKeyStatus.REVOKED
    assert deleted.id == "p-1"


async def test_asyncDeleteProject_unknownProject_raisesNotFound(admin_server: AdminServer) -> None:
    async with _async_admin(admin_server) as admin:
        with pytest.raises(NotFoundException):
            await admin.delete_project("missing")


async def test_asyncConstructor_withoutAnApiKey_raisesValueError() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        AsyncCyrockDbAdminClient()
