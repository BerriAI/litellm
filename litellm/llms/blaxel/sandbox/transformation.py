"""
Blaxel sandbox provider, over Blaxel's REST API (https://docs.blaxel.ai/api-reference).

The control plane creates and deletes the sandbox. Commands go to the sandbox's own URL.
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
    shell_command,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.secret_managers.main import get_secret_str

BLAXEL_API_BASE: Final = "https://api.blaxel.ai/v0"
BLAXEL_DEFAULT_IMAGE: Final = "blaxel/base-image:latest"
DEFAULT_SANDBOX_TIMEOUT: Final = 300
MIN_TTL_SECONDS: Final = 300
DEFAULT_MEMORY_MB: Final = 2048
MEMORY_MB_PER_VCPU: Final = 2048


class _Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True, frozen=True)

    name: str
    url: str


class _Created(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True, frozen=True)

    metadata: _Metadata


class _Process(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True, frozen=True)

    stdout: str = ""
    stderr: str = ""
    exitCode: int | None = None


class BlaxelSandboxConfig(BaseSandboxConfig):
    def validate_environment(
        self,
        api_key: str | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> str:
        key: Final = api_key or get_secret_str("BL_API_KEY")
        if not key:
            raise ValueError("Blaxel API key not set. Set BL_API_KEY or pass api_key=...")
        return key

    def _headers(self, container: ContainerHandle | str | None, api_key: str | None) -> Mapping[str, str]:
        hidden: Final[Mapping[str, object]] = container.hidden_params if isinstance(container, ContainerHandle) else {}
        key: Final = api_key or hidden.get("api_key") or self.validate_environment()
        workspace: Final = hidden.get("workspace") or get_secret_str("BL_WORKSPACE")
        return {"Authorization": f"Bearer {key}", **({"X-Blaxel-Workspace": str(workspace)} if workspace else {})}

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        memory_mb: int = DEFAULT_MEMORY_MB,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> ContainerHandle:
        if allow_internet_access is False:
            raise ValueError("The Blaxel sandbox provider cannot block internet access. Use another provider.")
        base: Final = api_base or BLAXEL_API_BASE
        headers: Final = self._headers(None, api_key)
        created_at: Final = time.time()
        response: Final = await self._http(client).post(
            url=join_api_base(base, "sandboxes"),
            headers=headers,
            json={
                "metadata": {"name": f"litellm-{uuid.uuid4().hex[:12]}"},
                "spec": {
                    "runtime": {
                        "image": template or BLAXEL_DEFAULT_IMAGE,
                        "memory": memory_mb,
                        "ttl": f"{max(timeout if timeout is not None else DEFAULT_SANDBOX_TIMEOUT, MIN_TTL_SECONDS)}s",
                    }
                },
            },
        )
        metadata: Final = _Created.model_validate(response.json()).metadata
        handle: Final = ContainerHandle(
            id=metadata.name,
            provider="blaxel",
            created_at=created_at,
            vcpus=memory_mb / MEMORY_MB_PER_VCPU,
            memory_gib=memory_mb / 1024,
        )
        handle.hidden_params = {
            "api_key": headers["Authorization"].removeprefix("Bearer "),
            "api_base": base,
            "workspace": headers.get("X-Blaxel-Workspace"),
            "url": metadata.url,
        }
        return handle

    async def arun_code(
        self,
        *,
        container: ContainerHandle | str,
        code: str,
        api_key: str | None = None,
        language: str = "python",
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> CodeExecutionResult:
        url: Final = container.hidden_params.get("url") if isinstance(container, ContainerHandle) else None
        if not url:
            raise ValueError(
                "Cannot run code from a sandbox name alone. Blaxel commands go to the sandbox's own URL, "
                "so pass the ContainerHandle returned by acreate_sandbox."
            )
        response: Final = await self._http(client).post(
            url=f"{str(url).rstrip('/')}/process",
            headers=self._headers(container, api_key),
            json={"command": shell_command(code, language), "waitForCompletion": True},
        )
        process: Final = _Process.model_validate(response.json())
        return self._shell_result(process.stdout, process.stderr, process.exitCode)

    async def adelete_sandbox(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> bool:
        hidden: Final[Mapping[str, object]] = container.hidden_params if isinstance(container, ContainerHandle) else {}
        base: Final = str(api_base or hidden.get("api_base") or BLAXEL_API_BASE)
        name: Final = container.id if isinstance(container, ContainerHandle) else container
        try:
            response: Final = await self._http(client).delete(
                url=join_api_base(base, f"sandboxes/{name}"), headers=self._headers(container, api_key)
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise
        return 200 <= response.status_code < 300
