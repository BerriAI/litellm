"""
Base Sandbox transformation configuration.

A sandbox provider runs an executable string inside an isolated container and
returns whatever the sandbox produced. The lifecycle is create container ->
run code -> delete container; `code_interpreter_tool` combines all three.
"""

import builtins
import shlex
import time
from functools import cache
from importlib.resources import files
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter

from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines
from litellm.types.llms.base import LiteLLMPydanticObjectBase

SANDBOX_MAX_OUTPUT_BYTES: Final = 10 * 1024 * 1024


class SandboxCost(LiteLLMPydanticObjectBase):
    """What a sandbox has cost so far. `cost_usd` is None when it cannot be known, never a guessed 0."""

    cost_usd: float | None = None
    source: Literal["provider", "estimate", "unknown"] = "unknown"
    seconds: float | None = None


class SandboxPrice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cost_per_vcpu_second: float = 0.0
    cost_per_gib_second: float = 0.0
    cost_per_disk_gib_second: float = 0.0


_SANDBOX_PRICES: Final = TypeAdapter(dict[str, SandboxPrice])


@cache
def get_sandbox_prices() -> dict[str, SandboxPrice]:
    return _SANDBOX_PRICES.validate_json(files("litellm").joinpath("sandbox_prices.json").read_bytes())


class ContainerHandle(LiteLLMPydanticObjectBase):
    """A live sandbox container. Carries everything needed to reach it again."""

    id: str
    provider: str
    domain: str | None = None
    created_at: float | None = None
    ended_at: float | None = None
    vcpus: float | None = None
    memory_gib: float | None = None
    disk_gib: float | None = None
    final_cost: SandboxCost | None = None

    model_config = {"extra": "allow"}

    _hidden_params: dict[str, builtins.object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params


def estimate_sandbox_cost(container: ContainerHandle | str, now: float | None = None) -> SandboxCost:
    """Price the sandbox's allocated resources over its lifetime from `sandbox_prices.json`."""
    if not isinstance(container, ContainerHandle) or container.created_at is None:
        return SandboxCost()
    end: Final = container.ended_at if container.ended_at is not None else (time.time() if now is None else now)
    seconds: Final = max(end - container.created_at, 0.0)
    price: Final = get_sandbox_prices().get(container.provider)
    if price is None or container.memory_gib is None or (price.cost_per_vcpu_second and container.vcpus is None):
        return SandboxCost(seconds=seconds)
    per_second: Final = (
        price.cost_per_vcpu_second * (container.vcpus or 0.0)
        + price.cost_per_gib_second * container.memory_gib
        + price.cost_per_disk_gib_second * (container.disk_gib or 0.0)
    )
    return SandboxCost(cost_usd=per_second * seconds, source="estimate", seconds=seconds)


def shell_command(code: str, language: str) -> str:
    if language == "bash":
        return code
    if language == "python":
        return f"python3 -c {shlex.quote(code)}"
    raise ValueError(f"Unsupported sandbox language: {language}. Use 'python' or 'bash'.")


class CodeExecutionResult(LiteLLMPydanticObjectBase):
    """Passthrough of the sandbox's own execution output."""

    stdout: str = ""
    stderr: str = ""
    results: list[dict[str, object]] = Field(default_factory=list)
    error: dict[str, object] | None = None
    execution_count: int | None = None
    object: str = "code_execution"

    model_config = {"extra": "allow"}

    _hidden_params: dict[str, builtins.object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params


class BaseSandboxConfig:
    """Provider-agnostic sandbox operations."""

    def validate_environment(self, api_key: str | None = None, **kwargs: builtins.object) -> str:
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
        container: ContainerHandle | str,
        code: str,
        api_key: str | None = None,
        **kwargs,
    ) -> CodeExecutionResult:
        raise NotImplementedError("arun_code must be implemented by provider")

    async def adelete_sandbox(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        **kwargs,
    ) -> bool:
        raise NotImplementedError("adelete_sandbox must be implemented by provider")

    async def aget_sandbox_cost(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        **kwargs,
    ) -> SandboxCost:
        return estimate_sandbox_cost(container)

    @staticmethod
    def _shell_result(stdout: str, stderr: str, exit_code: int | None) -> CodeExecutionResult:
        return CodeExecutionResult.model_validate(
            {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "error": None if exit_code == 0 else {"name": "ExitCode", "value": str(exit_code), "traceback": stderr},
            }
        )

    async def _read_capped_lines(self, response: httpx.Response) -> list[str]:
        lines: Final[list[str]] = []
        total = 0
        async for line in aiter_sse_lines(response.aiter_text()):
            total += len(line.encode("utf-8"))
            if total > SANDBOX_MAX_OUTPUT_BYTES:
                raise ValueError(
                    f"Sandbox output exceeded {SANDBOX_MAX_OUTPUT_BYTES} bytes; aborting to avoid unbounded memory use."
                )
            lines.append(line)
        return lines
