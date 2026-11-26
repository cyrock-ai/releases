"""The asyncio platform administration client.

The asyncio twin of :class:`~cyrock_db.CyrockDbAdminClient`: every operation, awaited rather than blocking, so an
asyncio application can administer projects and keys without stalling its event loop. Separately constructed and
closed, for the reason :mod:`cyrock_db.aio` gives for the data client. ``tests/test_admin_parity.py`` keeps the two
in step.

Java has no asynchronous admin client: there administration is a handful of blocking calls on any thread. Python
needs this one because a blocking call inside a coroutine stops every other coroutine on the loop.

**Thread-safe** in the asyncio sense: one instance serves a whole application, on its own loop.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import grpc

from . import _proto_converters as converters
from . import _proto_requests as requests
from ._admin import admin_options
from ._auth import AsyncAuthInterceptor, api_key_header
from ._proto import cyrock_db_platform_pb2_grpc as platform_grpc
from .exceptions import from_rpc_error
from .types import ApiKeyInfo, CreatedApiKey, NewApiKeyRequest, ProjectInfo

__all__ = ["AsyncCyrockDbAdminClient"]


class AsyncCyrockDbAdminClient:
    """An asyncio connection to the platform's administration services.

    Takes the same keyword arguments as :class:`~cyrock_db.CyrockDbAdminClient`, ``api_key`` required. Use as an
    async context manager, or await :meth:`close`::

        async with AsyncCyrockDbAdminClient(api_key=os.environ["CYROCK_DB_ADMIN_API_KEY"]) as admin:
            project = await admin.create_project(organization_id, "support")

    :raises TypeError: if ``consistent_reads`` is given
    :raises ValueError: if ``api_key`` is missing or blank, or another option is out of range
    """

    def __init__(self, **options: Any) -> None:
        self._options = admin_options(options)
        self._api_key = self._options.api_key or ""

        credentials  = self._options.credentials()
        arguments    = self._options.channel_arguments()
        interceptors = AsyncAuthInterceptor(self._metadata).interceptors()
        self._channel: grpc.aio.Channel = (
            grpc.aio.insecure_channel(self._options.target, options=arguments, interceptors=interceptors)
            if credentials is None
            else grpc.aio.secure_channel(
                self._options.target, credentials, options=arguments, interceptors=interceptors
            )
        )
        self._project_stub = platform_grpc.ProjectGrpcServiceStub(self._channel)  # type: ignore[no-untyped-call]
        self._api_key_stub = platform_grpc.ApiKeyGrpcServiceStub(self._channel)  # type: ignore[no-untyped-call]

    async def _metadata(self, _method: str) -> list[tuple[str, str]]:
        """The key, on every call: every administration service takes it."""
        return [api_key_header(self._api_key)]

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Closes the connection. Idempotent."""
        await self._channel.close()

    async def __aenter__(self) -> AsyncCyrockDbAdminClient:
        return self

    async def __aexit__(
        self,
        exc_type:  type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    # ─── Projects ───────────────────────────────────────────────────────────

    async def create_project(self, organization_id: str, name: str) -> ProjectInfo:
        """Asynchronous :meth:`~cyrock_db.CyrockDbAdminClient.create_project`.

        :raises NotFoundException: if the organization does not exist or is not the key's, which reads the same
        :raises ForbiddenException: if the key lacks the permission, or is pinned to a project: a pinned key acts only
            inside its own project
        :raises CyrockDbClientException: with status 400 if the name is blank, 409 if the organization already has a
            project of that name
        """
        return converters.to_project_info(
            await self._invoke(self._project_stub.Create, requests.create_project(organization_id, name))
        )

    async def list_projects(self) -> tuple[ProjectInfo, ...]:
        """Asynchronous :meth:`~cyrock_db.CyrockDbAdminClient.list_projects`.

        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return converters.to_project_infos(await self._invoke(self._project_stub.List, requests.empty()))

    async def delete_project(self, project_id: str) -> ProjectInfo:
        """Asynchronous :meth:`~cyrock_db.CyrockDbAdminClient.delete_project`.

        :raises NotFoundException: if there is no such project, or it is one the key may not reach (another
            organization's, or another project for a pinned key), which reads the same
        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return converters.to_project_info(
            await self._invoke(self._project_stub.Delete, requests.id_request(project_id))
        )

    # ─── API keys ───────────────────────────────────────────────────────────

    async def create_api_key(self, request: NewApiKeyRequest) -> CreatedApiKey:
        """Asynchronous :meth:`~cyrock_db.CyrockDbAdminClient.create_api_key`.

        :raises NotFoundException: if the organization, project or user does not exist, the user is not a member, or
            the organization is not the key's, which reads the same
        :raises ForbiddenException: if the key lacks the permission, the request grants more than allowed, or a key
            pinned to a project names another project or none
        :raises CyrockDbClientException: if the server returns another error
        """
        return converters.to_created_api_key(
            await self._invoke(self._api_key_stub.Create, requests.create_api_key(request))
        )

    async def list_api_keys(self) -> tuple[ApiKeyInfo, ...]:
        """Asynchronous :meth:`~cyrock_db.CyrockDbAdminClient.list_api_keys`.

        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return converters.to_api_key_infos(await self._invoke(self._api_key_stub.List, requests.empty()))

    async def revoke_api_key(self, api_key_id: str) -> ApiKeyInfo:
        """Asynchronous :meth:`~cyrock_db.CyrockDbAdminClient.revoke_api_key`.

        :raises NotFoundException: if there is no such key, or it is one the key may not reach (another
            organization's, or one outside a pinned key's project), which reads the same
        :raises ForbiddenException: if the key lacks the permission
        :raises CyrockDbClientException: if the server returns another error
        """
        return converters.to_api_key_info(
            await self._invoke(self._api_key_stub.Delete, requests.id_request(api_key_id))
        )

    # ─── Internals ──────────────────────────────────────────────────────────

    async def _invoke(self, rpc: Any, request: Any) -> Any:
        """Awaits one unary call, translating a gRPC failure into this SDK's exception type."""
        try:
            return await rpc(request, timeout=self._options.call_timeout)
        except grpc.aio.AioRpcError as error:
            raise from_rpc_error(error) from None
