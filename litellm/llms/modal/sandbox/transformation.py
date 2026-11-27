"""
Modal sandbox provider. Modal has no public REST API, so this goes through the `modal` SDK (`pip install 'litellm[modal]'`).
"""

import asyncio
import importlib
import time
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from types import ModuleType
from typing import Final, Protocol, TypeVar

from litellm.llms.base_llm.sandbox.transformation import (
    SANDBOX_MAX_OUTPUT_BYTES,
    BaseSandboxConfig,
    CodeExecutionResult,
    ContainerHandle,
    shell_command,
)
from litellm.secret_managers.main import get_secret_str

MODAL_APP_NAME: Final = "litellm-sandbox"
DEFAULT_SANDBOX_TIMEOUT: Final = 300
DEFAULT_CPUS: Final = 0.125
DEFAULT_MEMORY_MB: Final = 128


_T_co = TypeVar("_T_co", covariant=True)


class _Call(Protocol[_T_co]):
    def aio(self, *args: str, text: bool = True) -> Awaitable[_T_co]: ...


class _Stream(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...


class _Process(Protocol):
    stdout: _Stream
    stderr: _Stream
    wait: _Call[int]


class _ModalSandbox(Protocol):
    exec: _Call[_Process]
    terminate: _Call[None]


class ModalSandboxConfig(BaseSandboxConfig):
    def validate_environment(
        self,
        api_key: str | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> str:
        token_id: Final = get_secret_str("MODAL_TOKEN_ID")
        token_secret: Final = get_secret_str("MODAL_TOKEN_SECRET")
        key: Final = api_key or (f"{token_id}:{token_secret}" if token_id and token_secret else None)
        if not key or ":" not in key:
            raise ValueError(
                "Modal credentials not set. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, "
                "or pass api_key='<token_id>:<token_secret>'."
            )
        return key

    @asynccontextmanager
    async def _client(self, api_key: str | None) -> AsyncIterator[tuple[ModuleType, object]]:
        token_id, token_secret = self.validate_environment(api_key=api_key).split(":", 1)
        try:
            modal: Final = importlib.import_module("modal")
        except ImportError as e:
            raise ImportError(
                "The Modal sandbox provider needs the modal SDK. Run `pip install 'litellm[modal]'`."
            ) from e
        client: Final = await modal.Client.from_credentials.aio(token_id, token_secret)
        try:
            yield modal, client
        finally:
            await client.close.aio()

    @asynccontextmanager
    async def _sandbox(self, container: ContainerHandle | str, api_key: str | None) -> AsyncIterator[_ModalSandbox]:
        stored_key: Final = container.hidden_params.get("api_key") if isinstance(container, ContainerHandle) else None
        sandbox_id: Final = container.id if isinstance(container, ContainerHandle) else container
        async with self._client(api_key or (str(stored_key) if stored_key else None)) as (modal, client):
            yield await modal.Sandbox.from_id.aio(sandbox_id, client=client)

    @staticmethod
    async def _read_capped(stream: _Stream) -> bytes:
        buffer: Final = bytearray()
        async for chunk in stream:
            buffer.extend(chunk)
            if len(buffer) > SANDBOX_MAX_OUTPUT_BYTES:
                raise ValueError(
                    f"Sandbox output exceeded {SANDBOX_MAX_OUTPUT_BYTES} bytes; aborting to avoid unbounded memory use."
                )
        return bytes(buffer)

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        cpus: float = DEFAULT_CPUS,
        memory_mb: int = DEFAULT_MEMORY_MB,
        app_name: str | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> ContainerHandle:
        async with self._client(api_key) as (modal, client):
            app: Final = await modal.App.lookup.aio(
                app_name or get_secret_str("MODAL_APP_NAME") or MODAL_APP_NAME, create_if_missing=True, client=client
            )
            created_at: Final = time.time()
            ttl: Final = timeout if timeout is not None else DEFAULT_SANDBOX_TIMEOUT
            sandbox: Final = await modal.Sandbox.create.aio(
                app=app,
                image=modal.Image.from_registry(template) if template else modal.Image.debian_slim(),
                timeout=ttl,
                cpu=(cpus, cpus),
                memory=(memory_mb, memory_mb),
                block_network=allow_internet_access is False,
                client=client,
            )
        handle: Final = ContainerHandle(
            id=sandbox.object_id,
            provider="modal",
            created_at=created_at,
            expires_at=created_at + ttl,
            vcpus=cpus,
            memory_gib=memory_mb / 1024,
        )
        handle.hidden_params = {"api_key": self.validate_environment(api_key=api_key)}
        return handle

    async def arun_code(
        self,
        *,
        container: ContainerHandle | str,
        code: str,
        api_key: str | None = None,
        language: str = "python",
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> CodeExecutionResult:
        async with self._sandbox(container, api_key) as sandbox:
            process: Final = await sandbox.exec.aio("bash", "-c", shell_command(code, language), text=False)
            stdout, stderr = await asyncio.gather(self._read_capped(process.stdout), self._read_capped(process.stderr))
            return self._shell_result(
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                await process.wait.aio(),
            )

    async def adelete_sandbox(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> bool:
        async with self._sandbox(container, api_key) as sandbox:
            await sandbox.terminate.aio()
        return True
