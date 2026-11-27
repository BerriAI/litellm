"""
Tensorlake sandbox provider, over Tensorlake's REST API (https://docs.tensorlake.ai/api-reference/v2/introduction).

Commands run as sandbox processes: start one, poll until it exits, then read its stdout and stderr.
"""

import asyncio
import re
import time
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

TENSORLAKE_API_BASE: Final = "https://api.tensorlake.ai"
DEFAULT_SANDBOX_TIMEOUT: Final = 300
_SANDBOX_ID: Final = re.compile(r"[A-Za-z0-9_-]+")
DEFAULT_CPUS: Final = 1.0
DEFAULT_MEMORY_MB: Final = 1024
DEFAULT_DISK_MB: Final = 10240
PROCESS_POLL_SECONDS: Final = 0.25


class _Created(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True, frozen=True)

    sandbox_id: str
    sandbox_url: str


class _Process(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True, frozen=True)

    pid: int
    status: str
    exit_code: int | None = None


class _Output(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True, frozen=True)

    lines: tuple[str, ...] = ()


class TensorlakeSandboxConfig(BaseSandboxConfig):
    def validate_environment(
        self,
        api_key: str | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> str:
        key: Final = api_key or get_secret_str("TENSORLAKE_API_KEY")
        if not key:
            raise ValueError("Tensorlake API key not set. Set TENSORLAKE_API_KEY or pass api_key=...")
        return key

    def _key(self, container: ContainerHandle | str, api_key: str | None) -> str:
        hidden: Final[Mapping[str, object]] = container.hidden_params if isinstance(container, ContainerHandle) else {}
        return str(api_key or hidden.get("api_key") or self.validate_environment())

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        cpus: float = DEFAULT_CPUS,
        memory_mb: int = DEFAULT_MEMORY_MB,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> ContainerHandle:
        key: Final = self.validate_environment(api_key=api_key)
        base: Final = api_base or TENSORLAKE_API_BASE
        created_at: Final = time.time()
        ttl: Final = timeout if timeout is not None else DEFAULT_SANDBOX_TIMEOUT
        response: Final = await self._http(client).post(
            url=join_api_base(base, "sandboxes"),
            headers={"Authorization": f"Bearer {key}"},
            json={
                "timeout_secs": ttl,
                "resources": {"cpus": cpus, "memory_mb": memory_mb},
                **({"image": template} if template else {}),
                **({"network": {"allow_internet_access": False}} if allow_internet_access is False else {}),
            },
        )
        created: Final = _Created.model_validate(response.json())
        handle: Final = ContainerHandle(
            id=created.sandbox_id,
            provider="tensorlake",
            created_at=created_at,
            expires_at=created_at + ttl,
            vcpus=cpus,
            memory_gib=memory_mb / 1024,
            disk_gib=DEFAULT_DISK_MB / 1024,
        )
        handle.hidden_params = {"api_key": key, "api_base": base, "sandbox_url": created.sandbox_url}
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
        key: Final = self._key(container, api_key)
        if not isinstance(container, ContainerHandle) and not _SANDBOX_ID.fullmatch(container):
            raise ValueError("Invalid Tensorlake sandbox id.")
        sandbox_url: Final = (
            str(container.hidden_params.get("sandbox_url") or "") if isinstance(container, ContainerHandle) else ""
        ) or f"https://{container.id if isinstance(container, ContainerHandle) else container}.sandbox.tensorlake.ai"
        processes: Final = f"{sandbox_url.rstrip('/')}/api/v1/processes"
        headers: Final = {"Authorization": f"Bearer {key}"}
        http: Final = self._http(client)

        started: Final = await http.post(
            url=processes, headers=headers, json={"command": "bash", "args": ["-c", shell_command(code, language)]}
        )
        process = _Process.model_validate(started.json())  # rebind-ok: polled until the process exits
        expires_at: Final = container.expires_at if isinstance(container, ContainerHandle) else None
        while process.status == "running":
            if expires_at is not None and time.time() > expires_at:
                raise TimeoutError(f"Tensorlake process {process.pid} was still running when its sandbox timed out.")
            await asyncio.sleep(PROCESS_POLL_SECONDS)
            process = _Process.model_validate(
                (await http.get(url=f"{processes}/{process.pid}", headers=headers)).json()
            )

        stdout: Final = _Output.model_validate(
            (await http.get(url=f"{processes}/{process.pid}/stdout", headers=headers)).json()
        )
        stderr: Final = _Output.model_validate(
            (await http.get(url=f"{processes}/{process.pid}/stderr", headers=headers)).json()
        )
        return self._shell_result(_joined(stdout), _joined(stderr), process.exit_code, process.status)

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
        base: Final = str(api_base or hidden.get("api_base") or TENSORLAKE_API_BASE)
        sandbox_id: Final = container.id if isinstance(container, ContainerHandle) else container
        try:
            response: Final = await self._http(client).delete(
                url=join_api_base(base, f"sandboxes/{sandbox_id}"),
                headers={"Authorization": f"Bearer {self._key(container, api_key)}"},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise
        return 200 <= response.status_code < 300


def _joined(output: _Output) -> str:
    return "".join(f"{line}\n" for line in output.lines)
