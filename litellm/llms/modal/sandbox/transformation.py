"""
Modal sandbox provider. Modal has no public REST API, so this goes through the `modal` SDK (`pip install modal`).
"""

import time
from typing import Final

from litellm.llms.base_llm.sandbox.transformation import (
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


class ModalSandboxConfig(BaseSandboxConfig):
    def validate_environment(self, api_key: str | None = None, **kwargs: object) -> str:
        token_id: Final = get_secret_str("MODAL_TOKEN_ID")
        token_secret: Final = get_secret_str("MODAL_TOKEN_SECRET")
        key: Final = api_key or (f"{token_id}:{token_secret}" if token_id and token_secret else None)
        if not key or ":" not in key:
            raise ValueError(
                "Modal credentials not set. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, "
                "or pass api_key='<token_id>:<token_secret>'."
            )
        return key

    async def _client(self, api_key: str | None):
        try:
            import modal
        except ImportError as e:
            raise ImportError("The Modal sandbox provider needs the modal SDK. Run `pip install modal`.") from e
        token_id, token_secret = self.validate_environment(api_key=api_key).split(":", 1)
        return modal, await modal.Client.from_credentials.aio(token_id, token_secret)

    async def _sandbox(self, container: ContainerHandle | str, api_key: str | None):
        if isinstance(container, ContainerHandle) and container.hidden_params.get("sandbox") is not None:
            return container.hidden_params["sandbox"]
        modal, client = await self._client(
            api_key or (str(container.hidden_params["api_key"]) if isinstance(container, ContainerHandle) else None)
        )
        sandbox_id: Final = container.id if isinstance(container, ContainerHandle) else container
        return await modal.Sandbox.from_id.aio(sandbox_id, client=client)

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
        **kwargs: object,
    ) -> ContainerHandle:
        modal, client = await self._client(api_key)
        app: Final = await modal.App.lookup.aio(
            app_name or get_secret_str("MODAL_APP_NAME") or MODAL_APP_NAME, create_if_missing=True, client=client
        )
        created_at: Final = time.time()
        sandbox: Final = await modal.Sandbox.create.aio(
            app=app,
            image=modal.Image.from_registry(template) if template else modal.Image.debian_slim(),
            timeout=timeout if timeout is not None else DEFAULT_SANDBOX_TIMEOUT,
            cpu=cpus,
            memory=memory_mb,
            block_network=allow_internet_access is False,
            client=client,
        )
        handle: Final = ContainerHandle(
            id=sandbox.object_id, provider="modal", created_at=created_at, vcpus=cpus, memory_gib=memory_mb / 1024
        )
        handle.hidden_params = {"api_key": self.validate_environment(api_key=api_key), "sandbox": sandbox}
        return handle

    async def arun_code(
        self,
        *,
        container: ContainerHandle | str,
        code: str,
        api_key: str | None = None,
        language: str = "python",
        **kwargs: object,
    ) -> CodeExecutionResult:
        sandbox: Final = await self._sandbox(container, api_key)
        process: Final = await sandbox.exec.aio("bash", "-c", shell_command(code, language))
        stdout: Final = await process.stdout.read.aio()
        stderr: Final = await process.stderr.read.aio()
        return self._shell_result(stdout, stderr, await process.wait.aio())

    async def adelete_sandbox(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        **kwargs: object,
    ) -> bool:
        sandbox: Final = await self._sandbox(container, api_key)
        await sandbox.terminate.aio()
        return True
