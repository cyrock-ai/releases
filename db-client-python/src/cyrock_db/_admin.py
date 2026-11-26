"""The blocking platform administration client.

The Python face of ``CyrockDbAdminClient`` in client-java: the projects of an organization and the API keys that
reach them, over the platform's gRPC ``ProjectGrpcService`` and ``ApiKeyGrpcService``.

Kept apart from :class:`~cyrock_db.client.CyrockDbClient` on purpose: the data client works inside one project with a
key scoped to it, this one manages tenants and credentials with an organization-wide administrator's key. Every
call authenticates with the API key itself as ``x-api-key``: the platform refuses the short-lived token the data
client exchanges its key for, so there is no token here.

**Thread-safe.** One instance serves a whole application, as in Java.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar

import grpc

from . import _proto_converters as converters
from . import _proto_requests as requests
from ._auth import AuthInterceptor, api_key_header
from ._channel import ClientOptions
from ._proto import cyrock_db_platform_pb2_grpc as platform_grpc
from .exceptions import CyrockDbClientException, from_rpc_error
from .types import ApiKeyInfo, CreatedApiKey, NewApiKeyRequest, ProjectInfo

__all__ = ["CyrockDbAdminClient", "admin_options"]

_T = TypeVar("_T")


def admin_options(options: dict[str, Any]) -> ClientOptions:
    """The connection settings of an admin client: the data client's, minus ``consistent_reads``, with a key.

    :raises TypeError: if ``consistent_reads`` is given; the platform's administration services have no use for it
    :raises ValueError: if ``api_key`` is missing or blank
    """
    if "consistent_reads" in options:
        raise TypeError("consistent_reads is a data client option; the admin client does not take it")
    settings = ClientOptions(**options)
    if settings.api_key is None or not settings.api_key.strip():
        raise ValueError("api_key is required: the platform's administration services authenticate every call")
    return settings


class CyrockDbAdminClient:
    """A connection to the platform's administration services.

    Takes the data client's connection keyword arguments (``host``, ``port``, ``use_plaintext``,
    ``channel_credentials``, ``call_timeout``, ``retry_policy``, ``keepalive_time``, ``keepalive_timeout``,
    ``max_inbound_message_size``) with the same defaults, and requires ``api_key``: an organization-wide key, not
    pinned to a project, attributed to an ``ADMIN`` or ``OWNER`` of the organization and granted
    ``PROJECTS_CREATE``, ``PROJECTS_READ``, ``PROJECTS_DELETE``, ``API_KEYS_CREATE``, ``API_KEYS_LIST`` and
    ``API_KEYS_REVOKE``, plus every permission it is to hand out to new keys.

    Use as a context manager, or call :meth:`close` when finished::

        with CyrockDbAdminClient(api_key=os.environ["CYROCK_DB_ADMIN_API_KEY"]) as admin:
            project = admin.create_project(organization_id, "support")

    :raises TypeError: if ``consistent_reads`` is given
    :raises ValueError: if ``api_key`` is missing or blank, or another option is out of range
    """

    def __init__(self, **options: Any) -> None:
        self._options = admin_options(options)
        api_key       = self._options.api_key
        assert api_key is not None  # noqa: S101 - admin_options refuses a missing key

        credentials = self._options.credentials()
        arguments   = self._options.channel_arguments()
        self._raw_channel = (
            grpc.insecure_channel(self._options.target, options=arguments)
            if credentials is None
            else grpc.secure_channel(self._options.target, credentials, options=arguments)
        )
        # An interceptor rather than call credentials, for the reason given in _auth: grpc-python refuses call
        # credentials on the plaintext channel that is the default.
        channel = grpc.intercept_channel(self._raw_channel, AuthInterceptor(lambda _method: [api_key_header(api_key)]))

        self._project_stub = platform_grpc.ProjectGrpcServiceStub(channel)  # type: ignore[no-untyped-call]
        self._api_key_stub = platform_grpc.ApiKeyGrpcServiceStub(channel)  # type: ignore[no-untyped-call]

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Closes the connection. Idempotent."""
        self._raw_channel.close()

    def __enter__(self) -> CyrockDbAdminClient:
        return self

    def __exit__(
        self,
        exc_type:  type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ─── Projects ───────────────────────────────────────────────────────────

    def create_project(self, organization_id: str, name: str) -> ProjectInfo:
        """Creates a project in an organization. Needs ``PROJECTS_CREATE``.

        :raises NotFoundException: if the organization does not exist or is not the key's, which reads the same
        :raises ForbiddenException: if the key lacks the permission, or is pinned to a project: a pinned key acts only
            inside its own project
        :raises CyrockDbClientException: with status 400 if the name is blank, 409 if the organization already has a
            project of that name
        """
        return self._call(
            lambda: converters.to_project_info(
                self._project_stub.Create(
                    requests.create_project(organization_id, name), timeout=self._options.call_timeout
                )
            )
        )

    def list_projects(self) -> tuple[ProjectInfo, ...]:
        """Lists the projects the key can see: every project of its organization, or only its own for a pinned key.

        Needs ``PROJECTS_READ``. A read, so a ``retry_policy`` retries it.

        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return self._call(
            lambda: converters.to_project_infos(
                self._project_stub.List(requests.empty(), timeout=self._options.call_timeout)
            )
        )

    def delete_project(self, project_id: str) -> ProjectInfo:
        """Deletes a project and everything in it, the API keys pinned to it included. Not reversible.

        Needs ``PROJECTS_DELETE``.

        :raises NotFoundException: if there is no such project, or it is one the key may not reach (another
            organization's, or another project for a pinned key), which reads the same
        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return self._call(
            lambda: converters.to_project_info(
                self._project_stub.Delete(requests.id_request(project_id), timeout=self._options.call_timeout)
            )
        )

    # ─── API keys ───────────────────────────────────────────────────────────

    def create_api_key(self, request: NewApiKeyRequest) -> CreatedApiKey:
        """Creates an API key and returns it with its secret, which the platform shows this once.

        Needs ``API_KEYS_CREATE``. The new key cannot be granted a permission this client's key does not hold, nor,
        when it is attributed to a user, one that user does not hold where the key acts. Not retried by a default
        ``retry_policy``: a create whose answer was lost would leave a second key behind.

        :raises NotFoundException: if the organization, project or user does not exist, the user is not a member, or
            the organization is not the key's, which reads the same
        :raises ForbiddenException: if the key lacks the permission, the request grants more than allowed, or a key
            pinned to a project names another project or none
        :raises CyrockDbClientException: if the server returns another error
        """
        return self._call(
            lambda: converters.to_created_api_key(
                self._api_key_stub.Create(requests.create_api_key(request), timeout=self._options.call_timeout)
            )
        )

    def list_api_keys(self) -> tuple[ApiKeyInfo, ...]:
        """Lists the API keys the key can see, revoked ones included. Secrets are never listed.

        A key past its expiry is listed ``ACTIVE`` with its ``expires_at`` in the past; it authenticates nothing.

        Needs ``API_KEYS_LIST``. A read, so a ``retry_policy`` retries it.

        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return self._call(
            lambda: converters.to_api_key_infos(
                self._api_key_stub.List(requests.empty(), timeout=self._options.call_timeout)
            )
        )

    def revoke_api_key(self, api_key_id: str) -> ApiKeyInfo:
        """Revokes an API key: it authenticates nothing from now on, and stays listed as ``REVOKED``.

        Needs ``API_KEYS_REVOKE``. Idempotent: revoking a key that is already revoked changes nothing, so a
        ``retry_policy`` retries it. A key past its expiry is marked ``REVOKED`` like any other. A token the data
        client already exchanged with the key stays valid until it expires.

        :raises NotFoundException: if there is no such key, or it is one the key may not reach (another
            organization's, or one outside a pinned key's project), which reads the same
        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return self._call(
            lambda: converters.to_api_key_info(
                self._api_key_stub.Delete(requests.id_request(api_key_id), timeout=self._options.call_timeout)
            )
        )

    # ─── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _call(operation: Callable[[], _T]) -> _T:
        """Runs one call, translating a gRPC failure into this SDK's exception type, as the data client does."""
        try:
            return operation()
        except CyrockDbClientException:
            raise
        except grpc.RpcError as error:
            raise from_rpc_error(error) from None
