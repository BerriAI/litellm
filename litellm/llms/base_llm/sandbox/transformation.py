"""
Base Sandbox transformation configuration.

A sandbox provider runs an executable string inside an isolated container and
returns whatever the sandbox produced. The lifecycle is create container ->
run code -> delete container; `code_interpreter_tool` combines all three.
"""

from typing import Any, Union

import httpx

from pydantic import Field, PrivateAttr

from litellm.types.llms.base import LiteLLMPydanticObjectBase

SANDBOX_MAX_OUTPUT_BYTES = 10 * 1024 * 1024


class ContainerHandle(LiteLLMPydanticObjectBase):
    """A live sandbox container. Carries everything needed to reach it again."""

    id: str
    provider: str
    domain: str | None = None

    model_config = {"extra": "allow"}

    _hidden_params: dict = PrivateAttr(default_factory=dict)


class CodeExecutionResult(LiteLLMPydanticObjectBase):
    """Passthrough of the sandbox's own execution output."""

    stdout: str = ""
    stderr: str = ""
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    execution_count: int | None = None
    object: str = "code_execution"

    model_config = {"extra": "allow"}

    _hidden_params: dict = PrivateAttr(default_factory=dict)


class BaseSandboxConfig:
    """Provider-agnostic sandbox operations."""

    def validate_environment(self, api_key: str | None = None, **kwargs) -> str:
        raise NotImplementedError("validate_environment must be implemented by provider")

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> ContainerHandle:
        raise NotImplementedError("acreate_sandbox must be implemented by provider")

    async def arun_code(
        self,
        *,
        container: Union[ContainerHandle, str],
        code: str,
        api_key: str | None = None,
        **kwargs,
    ) -> CodeExecutionResult:
        raise NotImplementedError("arun_code must be implemented by provider")

    async def adelete_sandbox(
        self,
        *,
        container: Union[ContainerHandle, str],
        api_key: str | None = None,
        **kwargs,
    ) -> bool:
        raise NotImplementedError("adelete_sandbox must be implemented by provider")

    async def _read_capped_lines(self, response: httpx.Response) -> list[str]:
        lines: list[str] = []
        total = 0
        async for line in response.aiter_lines():
            total += len(line.encode("utf-8"))
            if total > SANDBOX_MAX_OUTPUT_BYTES:
                raise ValueError(
                    f"Sandbox output exceeded {SANDBOX_MAX_OUTPUT_BYTES} bytes; aborting to avoid unbounded memory use."
                )
            lines.append(line)
        return lines
