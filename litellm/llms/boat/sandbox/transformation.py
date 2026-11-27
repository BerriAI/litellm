"""
Boat sandbox provider, over Boat's REST API (https://docs.boat.dev/api/v1).

Boat reports dollars per sandbox, so cost comes from its usage endpoint instead of an estimate.
"""

import time
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.sandbox.transformation import (
    BaseSandboxConfig,
    CodeExecutionResult,
    ContainerHandle,
    SandboxCost,
    shell_command,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider

BOAT_API_BASE: Final = "https://boat.dev/api/v1"
BOAT_DEFAULT_TYPE: Final = "small"
DEFAULT_SANDBOX_TIMEOUT: Final = 300


class _Sandbox(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: str
    vcpu: float | None = None
    memoryGB: float | None = None


class _Created(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    sandbox: _Sandbox


class _Command(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    stdout: str = ""
    stderr: str = ""
    exitCode: int | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    dollars: float
    seconds: float | None = None


class BoatSandboxConfig(BaseSandboxConfig):
    def _http(self, client: AsyncHTTPHandler | None) -> AsyncHTTPHandler:
        return client or get_async_httpx_client(llm_provider=httpxSpecialProvider.Sandbox)

    def validate_environment(self, api_key: str | None = None, **kwargs: object) -> str:
        key: Final = api_key or get_secret_str("BOAT_API_KEY")
        if not key:
            raise ValueError("Boat API key not set. Set BOAT_API_KEY or pass api_key=...")
        return key

    def _auth(self, container: ContainerHandle | str, api_key: str | None, api_base: str | None) -> tuple[str, str]:
        hidden: Final[Mapping[str, object]] = container.hidden_params if isinstance(container, ContainerHandle) else {}
        key: Final = api_key or hidden.get("api_key") or self.validate_environment()
        return str(key), str(api_base or hidden.get("api_base") or BOAT_API_BASE)

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> ContainerHandle:
        key: Final = self.validate_environment(api_key=api_key)
        base: Final = api_base or BOAT_API_BASE
        created_at: Final = time.time()
        response: Final = await self._http(client).post(
            url=join_api_base(base, "sandboxes"),
            headers={"Authorization": f"Bearer {key}", "Idempotency-Key": uuid.uuid4().hex},
            json={
                "type": template or BOAT_DEFAULT_TYPE,
                "ttlSeconds": timeout if timeout is not None else DEFAULT_SANDBOX_TIMEOUT,
            },
        )
        sandbox: Final = _Created.model_validate(response.json()).sandbox
        handle: Final = ContainerHandle(
            id=sandbox.id, provider="boat", created_at=created_at, vcpus=sandbox.vcpu, memory_gib=sandbox.memoryGB
        )
        handle.hidden_params = {"api_key": key, "api_base": base}
        return handle

    async def arun_code(
        self,
        *,
        container: ContainerHandle | str,
        code: str,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str = "python",
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> CodeExecutionResult:
        key, base = self._auth(container, api_key, api_base)
        sandbox_id: Final = container.id if isinstance(container, ContainerHandle) else container
        response: Final = await self._http(client).post(
            url=join_api_base(base, f"sandboxes/{sandbox_id}/commands"),
            headers={"Authorization": f"Bearer {key}"},
            json={"command": shell_command(code, language)},
        )
        command: Final = _Command.model_validate(response.json())
        return self._shell_result(command.stdout, command.stderr, command.exitCode)

    async def adelete_sandbox(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> bool:
        key, base = self._auth(container, api_key, api_base)
        sandbox_id: Final = container.id if isinstance(container, ContainerHandle) else container
        try:
            response: Final = await self._http(client).delete(
                url=join_api_base(base, f"sandboxes/{sandbox_id}"),
                headers={"Authorization": f"Bearer {key}", "X-Ascii-Confirm-Delete": sandbox_id},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise
        return 200 <= response.status_code < 300

    async def aget_sandbox_cost(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> SandboxCost:
        key, base = self._auth(container, api_key, api_base)
        sandbox_id: Final = container.id if isinstance(container, ContainerHandle) else container
        response: Final = await self._http(client).get(
            url=join_api_base(base, f"sandboxes/{sandbox_id}/usage"),
            headers={"Authorization": f"Bearer {key}"},
        )
        usage: Final = _Usage.model_validate(response.json())
        return SandboxCost(cost_usd=usage.dollars, source="provider", seconds=usage.seconds)
